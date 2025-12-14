from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List

import torch

from hrl_lora.configs import DataCfg, LLPTrainCfg, PPOCfg, RewardCfg, TrainCfg
from hrl_lora.data.episode_loader import EpisodeChunkDataLoader, make_chunked_lerobot_dataset
from hrl_lora.models.router import RouterNetwork
from hrl_lora.models.moe_controller import MoEController
from hrl_lora.policy.llp_wrapper import LLPWrapper
from hrl_lora.rl.rollout import RolloutCollector
from hrl_lora.rl.ppo_agent import PPOAgent


def try_load_pi0(pretrained: str):
    """Try to load PI0Policy from the user's codebase.

    You will likely replace this with your own policy factory.
    """
    try:
        from common.policies.pi0.modeling_pi0 import PI0Policy  # type: ignore
    except Exception:
        try:
            from modeling_pi0 import PI0Policy  # type: ignore
        except Exception as e:
            raise ImportError(
                "Cannot import PI0Policy. Replace try_load_pi0() with your project's policy loader."
            ) from e

    policy = PI0Policy.from_pretrained(pretrained)
    return policy


def make_optimizer(params, cfg: LLPTrainCfg):
    return torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)


def warmup_train(
    *,
    policy,
    moe: MoEController,
    llp: LLPWrapper,
    dataset,
    num_experts: int,
    batch_size: int,
    epochs: int,
    device: str,
):
    """Warmup: train each expert on the same data for `epochs`.

    Note: Uses a simple shuffled frame sampler (not episode-ordered).
    """
    from torch.utils.data import DataLoader

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    for expert_id in range(num_experts):
        print(f"[warmup] expert={expert_id}")
        moe.set_active_expert(expert_id)
        moe.set_trainable_for_active_expert()

        for ep in range(epochs):
            for it, batch in enumerate(loader):
                # batch is a dict of tensors/lists already in batch-form
                loss, _ = llp.forward_loss(batch)
                llp.supervised_update_one_step(batch, precomputed_loss=loss)
                if it % 50 == 0:
                    print(f"  epoch={ep} it={it} loss={float(loss.detach().cpu().item()):.6f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_id", type=str, required=True)
    parser.add_argument("--root", type=str, default=None)
    parser.add_argument("--pretrained", type=str, required=True)
    parser.add_argument("--num_experts", type=int, default=4)
    parser.add_argument("--chunk_size", type=int, default=5)
    parser.add_argument("--episodes_per_update", type=int, default=32)
    parser.add_argument("--warmup_epochs", type=int, default=1)
    parser.add_argument("--warmup_batch", type=int, default=8)
    parser.add_argument("--total_updates", type=int, default=1000)
    parser.add_argument("--out_dir", type=str, default="./outputs/hrl_lora")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"

    # Load LLP policy (PI0/SmolVLA). Replace with your own loader if needed.
    policy = try_load_pi0(args.pretrained)
    policy.to(device)

    # Adapter names (expected to exist in your policy after LoRA injection)
    global_adapter = "global"
    expert_adapters = [f"expert_{i}" for i in range(args.num_experts)]

    moe = MoEController(policy, global_adapter=global_adapter, expert_adapters=expert_adapters)

    # Dataset for RL: image delta=[0], action delta=[0..chunk_size-1]
    image_keys: List[str] = getattr(policy.config, "image_features", []) or []
    if len(image_keys) == 0:
        raise ValueError("policy.config.image_features is empty. Provide image keys for embedding extraction.")

    dataset = make_chunked_lerobot_dataset(
        repo_id=args.repo_id,
        root=args.root,
        episodes=None,
        image_keys=image_keys,
        chunk_size=args.chunk_size,
    )

    # Sanity: chunk_size should match policy horizon
    if hasattr(policy.config, "n_action_steps") and int(policy.config.n_action_steps) != int(args.chunk_size):
        raise ValueError(
            f"chunk_size({args.chunk_size}) must match policy.config.n_action_steps({int(policy.config.n_action_steps)})."
        )

    # LLP optimizer (only trainable params are returned by MoEController)
    # During training, MoEController will toggle requires_grad each step.
    # Optimizer should own *all adapter params*; requires_grad will be toggled per-chunk.
    llp_optim = make_optimizer(moe.get_adapter_param_groups()[0]["params"], LLPTrainCfg())
    scaler = torch.cuda.amp.GradScaler() if (device != "cpu") else None
    llp = LLPWrapper(policy, llp_optim, scaler=scaler, amp=True, device=device)

    # Warmup
    warmup_train(
        policy=policy,
        moe=moe,
        llp=llp,
        dataset=dataset,
        num_experts=args.num_experts,
        batch_size=args.warmup_batch,
        epochs=args.warmup_epochs,
        device=device,
    )

    # Build router: infer obs_dim from one sample
    sample = dataset[0]
    # ensure batch form
    for k, v in list(sample.items()):
        if isinstance(v, torch.Tensor):
            sample[k] = v.unsqueeze(0)
        elif isinstance(v, str):
            sample[k] = [v]

    pooled = llp.extract_pooled_embed_first_frame(sample, image_keys=image_keys)
    obs_dim = int(pooled.shape[-1]) + 1

    router = RouterNetwork(obs_dim=obs_dim, num_experts=args.num_experts, hidden_dim=512).to(device)
    router_optim = torch.optim.AdamW(router.parameters(), lr=PPOCfg().lr)
    ppo = PPOAgent(router, router_optim, device=device)

    # RL episode iterator
    ep_loader = EpisodeChunkDataLoader(
        dataset=dataset,
        chunk_size=args.chunk_size,
        episodes_per_update=args.episodes_per_update,
        shuffle_episodes=True,
        seed=0,
    )

    reward_cfg = RewardCfg()
    collector = RolloutCollector(
        llp_wrapper=llp,
        moe_controller=moe,
        router=router,
        image_keys=image_keys,
        lambda_cons=reward_cfg.lambda_cons,
        reward_eps=reward_cfg.eps,
        device=device,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ep_iter = iter(ep_loader)
    epoch = 0
    for update_idx in range(args.total_updates):
        # Collect one PPO batch = episodes_per_update episodes.
        # Re-init iterator when exhausted, and reshuffle by advancing seed.
        try:
            batch_episodes = next(ep_iter)
        except StopIteration:
            epoch += 1
            ep_loader.seed = epoch
            ep_iter = iter(ep_loader)
            batch_episodes = next(ep_iter)
        buf = collector.collect(batch_episodes)
        roll = buf.finalize(gamma=PPOCfg().gamma, gae_lambda=PPOCfg().gae_lambda)
        stats = ppo.update(roll)

        if update_idx % 10 == 0:
            print(f"[update {update_idx}]", {k: round(v, 4) for k, v in stats.items()})

        # save router checkpoint occasionally
        if update_idx % 200 == 0:
            ckpt = {
                "router": router.state_dict(),
                "router_optim": router_optim.state_dict(),
                "update_idx": update_idx,
            }
            torch.save(ckpt, out_dir / f"router_{update_idx:06d}.pt")

    print("Done")


if __name__ == "__main__":
    main()
