from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DataCfg:
    repo_id: str
    root: str | None = None
    episodes: list[int] | None = None
    chunk_size: int = 5
    episodes_per_update: int = 32
    shuffle_episodes: bool = True
    seed: int = 0
    num_workers: int = 2
    pin_memory: bool = True


@dataclass
class RewardCfg:
    eps: float = 1e-8
    lambda_cons: float = 0.1


@dataclass
class PPOCfg:
    clip_eps: float = 0.2
    gamma: float = 0.99
    gae_lambda: float = 0.95
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    ppo_epochs: int = 4
    minibatch_size: int = 256
    lr: float = 3e-4
    max_grad_norm: float = 1.0


@dataclass
class LLPTrainCfg:
    lr: float = 1e-4
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    amp: bool = True


@dataclass
class TrainCfg:
    warmup_epochs: int = 1
    total_updates: int = 1000
    log_every: int = 10
    save_every: int = 200
    out_dir: str = "./outputs/hrl_lora"
