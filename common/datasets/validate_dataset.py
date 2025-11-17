# ==== 설정 ====
BASE = "/data/piper_multitask_push_new"  # 데이터셋 루트
N_EPISODES = 380                     # 총 episode 개수
REQUIRED_COLS = [
    "action", "action_joint",
    "observation.state", "observation.state_joint",
    "timestamp", "frame_index", "episode_index", "task_index",
]
DO_SAMPLE_SCAN = True   # HF datasets로 row 단위 None 검사 수행 여부

# ==== 코드 시작 ===
import logging
import os
import datetime
from pprint import pformat

import torch
import torch.distributed as dist

from common.datasets.make_dataloader import make_dataloader
from common.utils.random_utils import set_seed
from common.utils.utils import (
    get_safe_torch_device,
    init_logging,
    is_ddp_master
)
# adapter utils
from configs import parser
from configs.train import TrainPipelineConfig
from common.datasets.factory import make_dataset

@parser.wrap()
def train(cfg: TrainPipelineConfig):
    cfg.validate()

    dist_mode = getattr(cfg, "dist_mode", "none")  # 'ddp', 'fsdp', 'none'
    use_ddp = dist_mode == "ddp"
    use_fsdp = dist_mode == "fsdp"
    is_distributed = use_ddp or use_fsdp

    device = get_safe_torch_device(cfg.policy.device, log=True)
    torch.backends.cuda.matmul.allow_tf32 = True

    if is_distributed:
        if os.environ.get("LOCAL_RANK", -1) == -1:  # not called by torchrun, do not initialize dist.
            device, local_rank = torch.device("cuda"), 0  # single GPU
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl", timeout=datetime.timedelta(minutes=30))
        local_rank = dist.get_rank()
        device = torch.device("cuda", local_rank)
        torch.cuda.set_device(device)  # needed!
        logging.info(f"Local Rank ({local_rank}) Initialized for {dist_mode.upper()}")
    else:
        local_rank = 0
    cfg.policy.device = str(device)

    if is_ddp_master(is_distributed, local_rank):
        logging.info(pformat(cfg.to_dict()))

    if cfg.seed is not None:
        set_seed(cfg.seed)

    # --- Optional dataset debugging without touching make_dataloader ---
    if os.environ.get("DEBUG_DATASET", "0") == "1":
        logging.info("[DEBUG_DATASET] Running preflight dataset checks (no dataloader edits)")

        def _is_tensor_like(x):
            try:
                import torch
                return torch.is_tensor(x)
            except Exception:
                return False

        def _summarize_problem_sample(s):
            epi = s.get("episode_index", None)
            frm = s.get("frame_index", None)
            return f"(episode={epi}, frame={frm})"

        # 1) Item-level scan: ensure no required tensor fields are None
        REQUIRED_KEYS = [
            "observation.images.table",
            # Add more if your dataset uses them
            # "observation.images.wrist",
            # "observation.images.exo",
            "observation.state",
            "action",
        ]
        bad = []
        try:
            for i in range(len(make_dataset(cfg, split="train"))):
                try:
                    s = make_dataset(cfg, split="train")[i]
                    # Missing/None checks only on REQUIRED_KEYS that exist in sample schema
                    for k in REQUIRED_KEYS:
                        if k in s:
                            if s[k] is None:
                                bad.append((i, k, _summarize_problem_sample(s)))
                                if len(bad) <= 10:
                                    logging.warning(f"[DEBUG_DATASET] None in key='{k}' idx={i} {_summarize_problem_sample(s)}")
                        # If present and tensor-like, keep; if absent, we ignore here (some datasets may disable features)
                except Exception as e:
                    bad.append((i, "__exception__", str(e)))
                    if len(bad) <= 10:
                        logging.warning(f"[DEBUG_DATASET] Exception at idx={i}: {e}")
            if bad:
                logging.warning(f"[DEBUG_DATASET] Problem samples count: {len(bad)} (showing up to 10 above)")
            else:
                logging.info("[DEBUG_DATASET] Item-level scan passed: no None in REQUIRED_KEYS")
        except Exception as e:
            logging.warning(f"[DEBUG_DATASET] Item-level scan failed early: {e}")

        # 2) Batch-level probe with a local DataLoader using a debug collate_fn
        from torch.utils.data import DataLoader
        from torch.utils.data._utils.collate import default_collate

        def debug_collate(batch):
            try:
                return default_collate(batch)
            except Exception:
                if isinstance(batch[0], dict):
                    keys = list(batch[0].keys())
                    for k in keys:
                        vals = [b.get(k, None) for b in batch]
                        bad_pos = [i for i, v in enumerate(vals) if v is None]
                        if bad_pos:
                            epis = [batch[i].get("episode_index", None) for i in bad_pos]
                            frms = [batch[i].get("frame_index", None) for i in bad_pos]
                            logging.error(
                                f"[DEBUG_DATASET] None in key='{k}' at batch positions={bad_pos} "
                                f"(episode={epis}, frame={frms})"
                            )
                raise

        try:
            # Make a small, local dataloader that does not touch the global make_dataloader
            probe_bs = max(1, min(getattr(cfg, 'batch_size', 8), 32))
            probe_ds = make_dataset(cfg, split="train")
            probe_dl = DataLoader(
                probe_ds,
                batch_size=probe_bs,
                shuffle=False,
                num_workers=0,
                collate_fn=debug_collate,
                drop_last=False,
                pin_memory=False,
            )
            for _ in range(3):  # probe a few batches
                _ = next(iter(probe_dl))
            logging.info("[DEBUG_DATASET] Batch-level probe passed for first few batches")
        except Exception as e:
            logging.error(f"[DEBUG_DATASET] Batch-level probe found an issue: {e}")
            # You may choose to return early to inspect logs
            # return

    if is_ddp_master(is_distributed, local_rank):
        logging.info("Creating dataset")

    train_dataset = make_dataset(cfg, split="train")
    test_dataset = make_dataset(cfg, split="test")


    # create dataloader for offline training
    train_dataloader = make_dataloader(cfg, train_dataset, device)

    test_dataloader = make_dataloader(cfg, test_dataset, device)


if __name__ == "__main__":
    init_logging()
    train()
