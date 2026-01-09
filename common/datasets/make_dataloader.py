# common/datasets/make_dataloader.py
import torch
from torch.utils.data.distributed import DistributedSampler

from common.datasets.sampler import EpisodeAwareSampler, EpisodeAwareBatchSampler


def make_dataloader(cfg, dataset, device):
    """
    Stream-style(옵션 A)용:
    - cfg.dataloader_type == "stream" 이면 EpisodeAwareBatchSampler 사용
    - frame-level shuffle은 금지 (episode-level shuffle만 허용)
    """

    dataloader_type = getattr(cfg, "dataloader_type", "default")
    use_stream = dataloader_type in ("stream", "episode_stream")

    # drop_n_last_frames가 있으면 원래 코드가 EpisodeAwareSampler를 썼는데,
    # PCMB 목적이면 frame-level shuffle 금지 + batch_sampler로 교체.
    drop_n_last_frames = int(getattr(getattr(cfg, "policy", object()), "drop_n_last_frames", 0))

    # (선택) episode 앞쪽도 드랍하고 싶으면 cfg에 추가해서 사용
    drop_n_first_frames = int(getattr(getattr(cfg, "policy", object()), "drop_n_first_frames", 0))

    # 기본값: episode-level shuffle만
    shuffle_episodes = bool(getattr(cfg, "shuffle_episodes", True))

    # DDP/FSDP는 너가 안 쓴다고 했지만, 안전하게 기존 로직은 남겨둠
    dist_mode = getattr(cfg, "dist_mode", None)
    if dist_mode in ("ddp", "fsdp") and use_stream:
        raise ValueError(
            "Stream(EpisodeAwareBatchSampler) 모드에서는 현재 DDP/FSDP 분할 로직이 구현되어 있지 않습니다. "
            "단일 프로세스로 실행하거나, episode 단위 rank 분배를 추가 구현하세요."
        )

    if use_stream:
        batch_sampler = EpisodeAwareBatchSampler(
            dataset.episode_data_index,
            batch_size=cfg.batch_size,
            drop_n_first_frames=drop_n_first_frames,
            drop_n_last_frames=drop_n_last_frames,
            shuffle_episodes=shuffle_episodes,
            seed=int(getattr(cfg, "seed", 0)),
            allow_cross_episode_fill=False,  # 논문 stream은 true/ 일단 지금은 false로
            drop_last=False,
        )

        dataloader = torch.utils.data.DataLoader(
            dataset,
            num_workers=cfg.num_workers,
            batch_sampler=batch_sampler,  # batch_size/shuffle/sampler는 사용하지 않음
            pin_memory=device.type != "cpu",
            drop_last=False,
        )
        return dataloader

    # ---- 기존 default 경로 (PCMB 없이 일반 학습) ----
    if hasattr(cfg.policy, "drop_n_last_frames"):
        shuffle = False
        sampler = EpisodeAwareSampler(
            dataset.episode_data_index,
            drop_n_last_frames=cfg.policy.drop_n_last_frames,
            shuffle=True,  # (주의) frame-level shuffle
        )
    else:
        shuffle = True
        sampler = None

    if getattr(cfg, "dist_mode", "ddp") in ("ddp", "fsdp"):
        sampler = DistributedSampler(dataset, shuffle=shuffle, seed=cfg.seed, drop_last=False)
        shuffle = False

    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=cfg.num_workers,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        pin_memory=device.type != "cpu",
        drop_last=False,
    )
    return dataloader