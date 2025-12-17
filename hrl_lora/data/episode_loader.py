from __future__ import annotations

import random
from typing import Any, Dict, Iterator, List, Optional

import torch

from .chunk_types import ChunkBatch

# Try to import from the project; fallback to local copies if you're running this standalone in this sandbox.
try:
    from common.constants import ACTION, OBS_ROBOT
except Exception:
    ACTION = "action"
    OBS_ROBOT = "observation.state"

#try:
from common.datasets.lerobot_dataset import LeRobotDataset
#except Exception:
#    from lerobot_dataset_dataset import LeRobotDataset  # type: ignore

try:
    from common.datasets.utils import get_delta_indices
except Exception:
    from utils import get_delta_indices  # type: ignore


def _ensure_batch_dim(x: Any) -> Any:
    """Make sure tensors have a leading batch dim.

    - If x is Tensor and x.ndim==0: make (1,)
    - If x is Tensor and x.ndim>=1: unsqueeze(0)
    - If x is non-tensor (e.g., str): wrap into list of len 1
    """
    if isinstance(x, torch.Tensor):
        if x.ndim == 0:
            return x.view(1)
        return x.unsqueeze(0)
    # keep None / dicts as-is
    if x is None:
        return None
    if isinstance(x, (str, int, float)):
        return [x]
    return x


def make_chunked_lerobot_dataset(
    repo_id: str,
    root: Optional[str],
    episodes: Optional[List[int]],
    image_keys: List[str],
    router_stride: int,
    action_horizon: int,
    *,
    action_key: str = ACTION,
    state_key: str = OBS_ROBOT,
    download_videos: bool = True,
) -> LeRobotDataset:
    """Create a LeRobotDataset that returns (single) images + (action_horizon) actions via delta indices.

    Images: delta=[0]
    Actions: delta=[0..action_horizon-1] in frames (converted to timestamps)

    Note: router_stride is used by EpisodeChunkDataLoader to determine chunk anchor steps, not delta indices.
    """
    ds = LeRobotDataset(repo_id=repo_id, root=root, episodes=episodes, split="train", download_videos=download_videos)

    dt = 1.0 / float(ds.fps)
    # Delta timestamps in seconds
    action_deltas = [i * dt for i in range(action_horizon)]
    img_deltas = [0.0]

    # Only action uses a horizon. State should remain a single vector at the anchor index.
    delta_timestamps: Dict[str, List[float]] = {
        action_key: action_deltas,
    }

    # Only include image keys that actually exist in the dataset.
    present_img_keys = [k for k in image_keys if k in ds.meta.video_keys or k in ds.hf_dataset.column_names]
    for k in present_img_keys:
        delta_timestamps[k] = img_deltas

    ds.delta_timestamps = delta_timestamps
    ds.delta_indices = get_delta_indices(delta_timestamps, ds.fps)
    return ds


class EpisodeChunkDataLoader:
    """Iterate over episodes and yield chunks (time-ordered), grouped by episodes_per_update.

    Output:
      yield batch_episodes: List[List[ChunkBatch]]
        - outer list length == episodes_per_update (except possibly last)
        - each inner list is time-ordered chunks for that episode

    This is NOT a PyTorch DataLoader; it is a light iterator to preserve episode order.
    Chunks are yielded at steps spaced by router_stride.
    """

    def __init__(
        self,
        dataset: LeRobotDataset,
        router_stride: int,
        episodes_per_update: int,
        shuffle_episodes: bool = True,
        seed: int = 0,
        ddp_rank: int = 0,
        ddp_world_size: int = 1,
    ):
        self.dataset = dataset
        self.router_stride = router_stride
        self.episodes_per_update = episodes_per_update
        self.shuffle_episodes = shuffle_episodes
        self.seed = seed
        self.ddp_rank = ddp_rank
        self.ddp_world_size = ddp_world_size

        # Which absolute episode indices are available?
        if dataset.episodes is not None:
            self.episode_ids = list(dataset.episodes)
        else:
            # Fallback: assume 0..num_episodes-1
            self.episode_ids = list(range(dataset.num_episodes))

        # Shard episodes across DDP ranks
        self.episode_ids = self.episode_ids[self.ddp_rank :: self.ddp_world_size]

    def __iter__(self) -> Iterator[List[List[ChunkBatch]]]:
        rng = random.Random(self.seed)
        ep_ids = list(self.episode_ids)
        if self.shuffle_episodes:
            rng.shuffle(ep_ids)

        batch: List[List[ChunkBatch]] = []
        for ep_id in ep_ids:
            chunks = self._episode_to_chunks(ep_id)
            if len(chunks) == 0:
                continue
            batch.append(chunks)
            if len(batch) >= self.episodes_per_update:
                yield batch
                batch = []

        if batch:
            yield batch

    def _episode_to_chunks(self, ep_id: int) -> List[ChunkBatch]:
        # episode_data_index is indexed by absolute episode id
        ep_start = int(self.dataset.episode_data_index["from"][ep_id].item())
        ep_end = int(self.dataset.episode_data_index["to"][ep_id].item())
        ep_len = max(ep_end - ep_start, 0)
        if ep_len <= 0:
            return []

        action_horizon = len(self.dataset.delta_indices.get(ACTION, [0]))
        if action_horizon < 1:
            action_horizon = 1

        last_anchor_exclusive = ep_end - (action_horizon - 1)
        if last_anchor_exclusive <= ep_start:
            return []

        chunk_starts = list(range(ep_start, last_anchor_exclusive, self.router_stride))
        chunks: List[ChunkBatch] = []

        for i, idx in enumerate(chunk_starts):
            item: Dict[str, Any] = self.dataset[idx]

            # Convert to batch form (B=1)
            batch_item: Dict[str, Any] = {k: _ensure_batch_dim(v) for k, v in item.items()}

            # time progress scalar (B,)
            denom = max(ep_len - 1, 1)
            prog = torch.tensor([(idx - ep_start) / denom], dtype=torch.float32)

            # done flag (B,)
            done = torch.tensor([i == (len(chunk_starts) - 1)], dtype=torch.bool)

            meta = {
                "episode_id": ep_id,
                "ep_start": ep_start,
                "ep_end": ep_end,
                "anchor_index": idx,
                "chunk_idx": i,
                "router_stride": self.router_stride,
                "action_horizon": action_horizon,
            }

            chunks.append(ChunkBatch(batch=batch_item, time_progress=prog, done=done, meta=meta))

        return chunks
