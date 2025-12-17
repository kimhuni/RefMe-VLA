from __future__ import annotations
""" mango
CUDA_VISIBLE_DEVICES=1 python -m hrl_lora.train_hrl_lora \
  --repo_id /data/piper_pickplace/lerobot_5hz \
  --pretrained /ckpt/pi0 \
  --out_dir /result/ghkim/rl_lora/piper_pickplace \
  --hrl.chunk_size 5 --hrl.episodes_per_update 32 --hrl.lambda_cons 0.02 --ppo_ent_coef 0.01\ # 0.1 0.01
  --warmup_epochs 1 \
  --wandb --wandb_project RefMe --wandb_run_name HRL-LoRA_pickplace_lambda_0.02_ent_0.01 \
  --wandb_tags piper,debug
"""

""" sushi
CUDA_VISIBLE_DEVICES=2 python -m hrl_lora.train_hrl_lora \
  --repo_id "/data/ghkim/press the blue button four times/lerobot_5hz" \
  --pretrained /ckpt/pi0 \
  --out_dir /result/ghkim/rl_lora/piper_press_four_times \
  --hrl.chunk_size 5 --hrl.episodes_per_update 32 --hrl.lambda_cons 0.1 --ppo_ent_coef 0.05 \
  --warmup_epochs 1 \
  --wandb --wandb_project RefMe --wandb_run_name HRL-LoRA_press_four_times_lambda_0.1_ent_0.05 \
  --wandb_tags piper,debug
"""


import argparse
import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import draccus
import torch
import os

try:
    import wandb
except Exception:
    wandb = None

# ---- project configs (existing) ----
from configs.train import TrainPipelineConfig
from configs.default import DatasetConfig
from configs.policies import PreTrainedConfig

# ---- existing project factories ----
from common.datasets.factory import make_dataset
from common.policies.factory import make_policy

# ---- HRL-LoRA modules (your project) ----
from hrl_lora.configs.configs import LLPTrainCfg, PPOCfg, RewardCfg
from hrl_lora.data.episode_loader import EpisodeChunkDataLoader, make_chunked_lerobot_dataset
from hrl_lora.models.router import RouterNetwork
from hrl_lora.policy.llp_wrapper import LLPWrapper
from hrl_lora.rl.rollout import RolloutCollector
from hrl_lora.rl.ppo_agent import PPOAgent

# ---- NEW: MoE LoRA (in-place) ----
from hrl_lora.configs.moe_config import MoELoRAConfig
from hrl_lora.models.inject_lora_moe import inject_lora_moe_inplace
from hrl_lora.models.moe_controller import MoEController


# --------------------------------------------------------------------------------------
# HRL-only knobs: keep your --hrl.* CLI style
# --------------------------------------------------------------------------------------
@dataclass
class HRLConfig:
    chunk_size: int = 5
    episodes_per_update: int = 32
    lambda_cons: float = 0.1


@dataclass
class HRLArgs:
    hrl: HRLConfig = field(default_factory=HRLConfig)


def parse_hrl(argv=None) -> HRLConfig:
    parsed: HRLArgs = draccus.parse(HRLArgs, args=argv)
    return parsed.hrl


# --------------------------------------------------------------------------------------
# Build minimal TrainPipelineConfig (reusing existing configs, but only what we need)
# --------------------------------------------------------------------------------------
def _build_dataset_cfg(repo_id_arg: str) -> DatasetConfig:
    kwargs = {}
    for f in dataclasses.fields(DatasetConfig):
        if f.name == "repo_id":
            kwargs["repo_id"] = repo_id_arg
        elif f.name == "root":
            # 너가 원한 정책: root에도 repo_id를 그대로 넣는다
            kwargs["root"] = repo_id_arg
        elif f.name == "split":
            kwargs["split"] = "train"
    return DatasetConfig(**kwargs)  # type: ignore[arg-type]


def build_pipeline_cfg(*, repo_id_arg: str, pretrained: str, out_dir: str) -> TrainPipelineConfig:
    train_ds = _build_dataset_cfg(repo_id_arg)

    policy_cfg = PreTrainedConfig.from_pretrained(pretrained)
    policy_cfg.pretrained_path = pretrained

    cfg = TrainPipelineConfig(train_dataset=train_ds)
    cfg.policy = policy_cfg
    cfg.output_dir = Path(out_dir)
    return cfg


# --------------------------------------------------------------------------------------
# Optim
# --------------------------------------------------------------------------------------
def make_optimizer(params, cfg: LLPTrainCfg):
    return torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)


# --------------------------------------------------------------------------------------
# Warmup (expert별로 동일 데이터로 학습)
# --------------------------------------------------------------------------------------
def warmup_train(
    *,
    moe: MoEController,
    llp: LLPWrapper,
    dataset,
    num_experts: int,
    batch_size: int,
    epochs: int,
    num_workers: int,
    pin_memory: bool,
    run=None,
    global_step: int = 0,
) -> int:
    from torch.utils.data import DataLoader

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    for expert_id in range(num_experts):
        print(f"[warmup] expert={expert_id}")
        moe.set_active_expert(expert_id)
        moe.set_trainable_for_active_expert()

        for ep in range(epochs):
            for it, batch in enumerate(loader):
                loss, _ = llp.forward_loss(batch)
                llp.supervised_update_one_step(batch, precomputed_loss=loss)
                if it % 50 == 0:
                    print(f"  epoch={ep} it={it} loss={float(loss.detach().cpu().item()):.6f}")
                if run is not None:
                    run.log(
                        {
                            "warmup/loss": float(loss.detach().cpu().item()),
                            "warmup/expert": expert_id,
                            "warmup/epoch": ep,
                            "warmup/iter": it,
                        },
                        step=global_step,
                    )
                global_step += 1

    return global_step


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--repo_id", type=str, required=True)
    parser.add_argument("--pretrained", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_experts", type=int, default=4)

    # warmup / training
    parser.add_argument("--warmup_epochs", type=int, default=1)
    parser.add_argument("--warmup_batch", type=int, default=8)
    parser.add_argument("--total_updates", type=int, default=1000)

    # NOTE: keep legacy flags too (optional)
    parser.add_argument("--chunk_size", type=int, default=5)
    parser.add_argument("--episodes_per_update", type=int, default=32)

    # dataloader / perf
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--pin_memory", action="store_true")

    # LLP optimization
    parser.add_argument("--llp_lr", type=float, default=3e-4)
    parser.add_argument("--llp_weight_decay", type=float, default=0.0)

    # Router/PPO optimization
    parser.add_argument("--router_lr", type=float, default=3e-4)
    parser.add_argument("--ppo_clip_eps", type=float, default=0.2)
    parser.add_argument("--ppo_vf_coef", type=float, default=0.5)
    parser.add_argument("--ppo_ent_coef", type=float, default=0.01)
    parser.add_argument("--ppo_gamma", type=float, default=0.99)
    parser.add_argument("--ppo_gae_lambda", type=float, default=0.95)
    parser.add_argument("--ppo_epochs", type=int, default=4)
    parser.add_argument("--ppo_minibatch", type=int, default=256)

    # Reward shaping
    parser.add_argument("--lambda_cons", type=float, default=None, help="Override --hrl.lambda_cons if set")

    # MoE LoRA injection
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.0)

    # W&B
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="hrl-moe-lora")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--wandb_tags", type=str, default="")

    args, unknown = parser.parse_known_args()

    device = args.device if (torch.cuda.is_available() or args.device == "cpu") else "cpu"

    # HRL knobs from --hrl.*
    hrl = parse_hrl(unknown)
    router_stride = int(hrl.chunk_size)
    episodes_per_update = int(hrl.episodes_per_update)

    if args.lambda_cons is not None:
        hrl.lambda_cons = float(args.lambda_cons)

    use_wandb = bool(args.wandb)
    if use_wandb and wandb is None:
        raise RuntimeError("wandb is not installed but --wandb was set. Please `pip install wandb`.")

    run = None
    if use_wandb:
        tags = [t for t in args.wandb_tags.split(",") if t.strip()]
        run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name,
            tags=tags,
            config={
                "repo_id": args.repo_id,
                "pretrained": args.pretrained,
                "num_experts": args.num_experts,
                "device": device,
                "hrl": dataclasses.asdict(hrl),
                "warmup_epochs": args.warmup_epochs,
                "warmup_batch": args.warmup_batch,
                "total_updates": args.total_updates,
                "router_stride": router_stride,
                "episodes_per_update": episodes_per_update,
                "llp_lr": args.llp_lr,
                "llp_weight_decay": args.llp_weight_decay,
                "router_lr": args.router_lr,
                "ppo": {
                    "clip_eps": args.ppo_clip_eps,
                    "vf_coef": args.ppo_vf_coef,
                    "ent_coef": args.ppo_ent_coef,
                    "gamma": args.ppo_gamma,
                    "gae_lambda": args.ppo_gae_lambda,
                    "epochs": args.ppo_epochs,
                    "minibatch": args.ppo_minibatch,
                },
                "lora": {"r": args.lora_r, "alpha": args.lora_alpha, "dropout": args.lora_dropout},
            },
        )

    # Build minimal cfg for dataset + policy factory
    cfg = build_pipeline_cfg(repo_id_arg=args.repo_id, pretrained=args.pretrained, out_dir=args.out_dir)

    # 1) Dataset(meta) + Policy
    train_dataset = make_dataset(cfg, split="train")
    policy = make_policy(cfg=cfg.policy, ds_meta=train_dataset.meta)
    policy.to(device)

    print("input_features:", policy.config.input_features)
    print("output_features:", policy.config.output_features)

    # 2) Inject MoE LoRA IN-PLACE (LLM q_proj/v_proj only)
    moe_cfg = MoELoRAConfig(
        num_experts=args.num_experts,
        r=args.lora_r,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
        target_modules=("q_proj", "v_proj"),
        include_name_substrings=("language_model", "gemma_expert"),  # LLM 쪽에만 제한
    )
    replaced = inject_lora_moe_inplace(policy, moe_cfg)
    print(f"[moe] replaced {len(replaced)} modules. e.g. {replaced[:5]}")

    # 3) MoE controller + LLP wrapper
    moe = MoEController(policy)

    llp_cfg = LLPTrainCfg()
    llp_cfg.lr = float(args.llp_lr)
    llp_cfg.weight_decay = float(args.llp_weight_decay)
    llp_optim = make_optimizer(moe.get_adapter_param_groups()[0]["params"], llp_cfg)
    scaler = torch.cuda.amp.GradScaler() if (device != "cpu") else None
    llp = LLPWrapper(policy, llp_optim, scaler=scaler, amp=True, device=device)

    # 4) RL Dataset (episode-ordered chunks)
    image_features = getattr(policy.config, "image_features", {}) or {}
    image_keys: List[str] = list(image_features.keys())
    if len(image_keys) == 0:
        raise ValueError("policy.config.image_features is empty. Provide image keys for embedding extraction.")

    root = getattr(cfg.train_dataset, "root", None)
    action_horizon = int(policy.config.n_action_steps)
    print(f"[HRL] router_stride={router_stride}, action_horizon={action_horizon}")

    rl_dataset = make_chunked_lerobot_dataset(
        repo_id=args.repo_id,
        root=root,
        episodes=None,
        image_keys=image_keys,
        router_stride=router_stride,
        action_horizon=action_horizon,
    )

    print("policy.config has method?", hasattr(policy.config, "method"))
    print("policy.config.method =", getattr(policy.config, "method", "MISSING"))
    print("policy.config keys =", list(getattr(policy.config, "to_dict", lambda: {})().keys())[:50])

    # 5) Warmup
    global_step = 0
    global_step = warmup_train(
        moe=moe,
        llp=llp,
        dataset=rl_dataset,
        num_experts=args.num_experts,
        batch_size=args.warmup_batch,
        epochs=args.warmup_epochs,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        run=run,
        global_step=global_step,
    )

    # 6) Router init
    sample = rl_dataset[0]
    for k, v in list(sample.items()):
        if isinstance(v, torch.Tensor):
            sample[k] = v.unsqueeze(0)
        elif isinstance(v, str):
            sample[k] = [v]

    pooled = llp.extract_pooled_embed_first_frame(sample, image_keys=image_keys)
    obs_dim = int(pooled.shape[-1]) + 1  # + time scalar

    router = RouterNetwork(obs_dim=obs_dim, num_experts=args.num_experts, hidden_dim=512).to(device)
    router_optim = torch.optim.AdamW(router.parameters(), lr=float(args.router_lr))

    ppo_cfg = PPOCfg()
    ppo_cfg.clip_eps = float(args.ppo_clip_eps)
    ppo_cfg.vf_coef = float(args.ppo_vf_coef)
    ppo_cfg.ent_coef = float(args.ppo_ent_coef)
    ppo_cfg.gamma = float(args.ppo_gamma)
    ppo_cfg.gae_lambda = float(args.ppo_gae_lambda)
    ppo_cfg.epochs = int(args.ppo_epochs)
    ppo_cfg.minibatch_size = int(args.ppo_minibatch)

    ppo = PPOAgent(
        router,
        router_optim,
        clip_eps=float(args.ppo_clip_eps),
        value_coef=float(args.ppo_vf_coef),
        entropy_coef=float(args.ppo_ent_coef),
        gamma=float(args.ppo_gamma),
        gae_lambda=float(args.ppo_gae_lambda),
        ppo_epochs=int(args.ppo_epochs),
        minibatch_size=int(args.ppo_minibatch),
        device=device,
    )

    # 7) Episode loader
    ep_loader = EpisodeChunkDataLoader(
        dataset=rl_dataset,
        router_stride=router_stride,
        episodes_per_update=episodes_per_update,
        shuffle_episodes=True,
        seed=0,
    )

    # Use lambda_cons from HRL config
    collector = RolloutCollector(
        llp_wrapper=llp,
        moe_controller=moe,
        router=router,
        image_keys=image_keys,
        lambda_cons=float(hrl.lambda_cons),
        reward_eps=RewardCfg().eps,
        device=device,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 8) PPO loop
    ep_iter = iter(ep_loader)
    epoch = 0
    for update_idx in range(args.total_updates):
        try:
            batch_episodes = next(ep_iter)
        except StopIteration:
            epoch += 1
            ep_loader.seed = epoch
            ep_iter = iter(ep_loader)
            batch_episodes = next(ep_iter)

        buf = collector.collect(batch_episodes)
        print("len step_experts:", len(getattr(buf, "step_experts", [])))
        print("num episodes:", len(getattr(buf, "episode_experts", [])))

        roll = buf.finalize(gamma=float(args.ppo_gamma), gae_lambda=float(args.ppo_gae_lambda))
        stats = ppo.update(roll)

        if run is not None:
            logd = {f"ppo/{k}": float(v) for k, v in stats.items()}
            # ---- routing diagnostics ----
            logd["routing/switch_rate"] = float(getattr(buf, "switch_rate", 0.0))

            # expert histogram (requires wandb)
            if hasattr(buf, "step_experts") and len(buf.step_experts) > 0:
                logd["routing/expert_histogram"] = wandb.Histogram(buf.step_experts)

            # ---- routing distribution as scalar fractions (auto line-plots) ----
            if hasattr(buf, "step_experts") and len(buf.step_experts) > 0:
                counts = [0] * args.num_experts
                for e in buf.step_experts:
                    if 0 <= int(e) < args.num_experts:
                        counts[int(e)] += 1
                tot = max(1, sum(counts))
                for i, c in enumerate(counts):
                    logd[f"routing/expert_count_{i}"] = int(c)
                    logd[f"routing/expert_frac_{i}"] = float(c / tot)

                # optional: how peaked the distribution is (0=uniform, 1=all-in-one)
                max_frac = max(c / tot for c in counts)
                logd["routing/expert_max_frac"] = float(max_frac)

            # Try to log rollout-level signals if present
            for key in ["mean_reward", "reward_mean", "avg_reward", "episode_reward_mean"]:
                if hasattr(buf, key):
                    logd[f"rollout/{key}"] = float(getattr(buf, key))
            run.log(logd, step=global_step)

        # episode-level routing heatmap (log occasionally)
            if (update_idx % 20 == 0) and hasattr(buf, "episode_experts") and len(buf.episode_experts) > 0:
                import numpy as np
                import matplotlib.pyplot as plt

                seqs = buf.episode_experts
                max_len = max(len(s) for s in seqs)
                mat = -1 * np.ones((len(seqs), max_len), dtype=np.int32)
                for i, s in enumerate(seqs):
                    mat[i, : len(s)] = np.asarray(s, dtype=np.int32)

                fig, ax = plt.subplots()
                ax.imshow(mat, aspect="auto")
                ax.set_title("Expert routing per episode (rows=episodes, cols=chunks)")
                ax.set_xlabel("chunk index")
                ax.set_ylabel("episode index")
                logd["routing/episode_expert_map"] = wandb.Image(fig)
                plt.close(fig)
                run.log({"routing/episode_expert_map": logd["routing/episode_expert_map"]}, step=global_step)

        global_step += 1

        if update_idx % 10 == 0:
            print(f"[update {update_idx}]", {k: round(v, 4) for k, v in stats.items()})

        if update_idx % 200 == 0:
            ckpt = {
                "router": router.state_dict(),
                "router_optim": router_optim.state_dict(),
                "update_idx": update_idx,
            }
            torch.save(ckpt, out_dir / f"router_{update_idx:06d}.pt")

    if run is not None:
        run.finish()

    print("Done")


if __name__ == "__main__":
    main()