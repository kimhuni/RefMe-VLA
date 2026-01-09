#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Iterator, Union, List, Optional, Sequence, Tuple

import torch


class EpisodeAwareSampler:
    def __init__(
        self,
        episode_data_index: dict,
        episode_indices_to_use: Union[list, None] = None,
        drop_n_first_frames: int = 0,
        drop_n_last_frames: int = 0,
        shuffle: bool = False,
    ):
        """Sampler that optionally incorporates episode boundary information.

        Args:
            episode_data_index: Dictionary with keys 'from' and 'to' containing the start and end indices of each episode.
            episode_indices_to_use: List of episode indices to use. If None, all episodes are used.
                                    Assumes that episodes are indexed from 0 to N-1.
            drop_n_first_frames: Number of frames to drop from the start of each episode.
            drop_n_last_frames: Number of frames to drop from the end of each episode.
            shuffle: Whether to shuffle the indices.
        """
        indices = []
        for episode_idx, (start_index, end_index) in enumerate(
            zip(episode_data_index["from"], episode_data_index["to"], strict=True)
        ):
            if episode_indices_to_use is None or episode_idx in episode_indices_to_use:
                indices.extend(
                    range(start_index.item() + drop_n_first_frames, end_index.item() - drop_n_last_frames)
                )

        self.indices = indices
        self.shuffle = shuffle

    def __iter__(self) -> Iterator[int]:
        if self.shuffle:
            for i in torch.randperm(len(self.indices)):
                yield self.indices[i]
        else:
            for i in self.indices:
                yield i

    def __len__(self) -> int:
        return len(self.indices)

class EpisodeAwareBatchSampler:
    """
    MemoryVLA(B.3)의 'streaming queue' 동작을 PyTorch BatchSampler로 구현.

    - episode 내부 frame은 순서 유지
    - batch는 가능한 한 single-episode로 구성
    - episode 끝에서 batch가 덜 차면, 다음 episode frame으로 남은 슬롯 채움
    - (중요) frame-level shuffle 금지. 대신 episode-level shuffle만 지원.

    반환: List[int] (DataLoader가 dataset[idx]로 접근해 batch를 구성)
    """

    def __init__(
        self,
        episode_data_index: dict,
        batch_size: int,
        *,
        episode_indices_to_use: Optional[Sequence[int]] = None,
        drop_n_first_frames: int = 0,
        drop_n_last_frames: int = 0,
        shuffle_episodes: bool = True,
        seed: int = 0,
        allow_cross_episode_fill: bool = True,
        drop_last: bool = False,
    ):
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")

        self.batch_size = int(batch_size)
        self.drop_last = bool(drop_last)
        self.allow_cross_episode_fill = bool(allow_cross_episode_fill)

        # Build per-episode index ranges (already in time order)
        all_eps: List[int] = list(range(len(episode_data_index["from"])))
        if episode_indices_to_use is not None:
            use_set = set(int(x) for x in episode_indices_to_use)
            all_eps = [e for e in all_eps if e in use_set]

        self.episodes: List[List[int]] = []
        for e in all_eps:
            s = int(episode_data_index["from"][e].item()) + int(drop_n_first_frames)
            t = int(episode_data_index["to"][e].item()) - int(drop_n_last_frames)
            if t > s:
                self.episodes.append(list(range(s, t)))

        self.shuffle_episodes = bool(shuffle_episodes)
        self.seed = int(seed)

        # length는 "대략"이 아니라 정확히 세려면 cross-episode fill까지 고려해야 하는데,
        # DataLoader에서 __len__은 진행률 표시용이라 안전하게 upper bound로 둠.
        self._num_frames = sum(len(ep) for ep in self.episodes)

    def __iter__(self) -> Iterator[List[int]]:
        if len(self.episodes) == 0:
            return
            yield  # for mypy

        # Episode order shuffle (no frame shuffle)
        if self.shuffle_episodes:
            g = torch.Generator()
            g.manual_seed(self.seed)
            order = torch.randperm(len(self.episodes), generator=g).tolist()
        else:
            order = list(range(len(self.episodes)))

        ep_ptr = 0
        frame_ptr = 0

        while ep_ptr < len(order):
            ep = self.episodes[order[ep_ptr]]
            # episode 끝이면 다음 episode로
            if frame_ptr >= len(ep):
                ep_ptr += 1
                frame_ptr = 0
                continue

            batch: List[int] = []

            # (1) 가능한 한 현재 episode에서 채움
            take = min(self.batch_size, len(ep) - frame_ptr)
            batch.extend(ep[frame_ptr : frame_ptr + take])
            frame_ptr += take

            # (2) episode가 끝나서 batch가 덜 찼으면, 다음 episode로 fill (논문 stream과 동일)
            if self.allow_cross_episode_fill and len(batch) < self.batch_size:
                need = self.batch_size - len(batch)
                # 다음 episode들에서 연속으로 가져와 채움
                next_ep_ptr = ep_ptr + 1
                while need > 0 and next_ep_ptr < len(order):
                    next_ep = self.episodes[order[next_ep_ptr]]
                    if len(next_ep) == 0:
                        next_ep_ptr += 1
                        continue
                    take2 = min(need, len(next_ep))
                    batch.extend(next_ep[:take2])
                    need -= take2
                    # 다음 ep에서 일부를 가져갔으면, 그 ep의 frame_ptr을 그만큼 진행시키기 위해
                    # ep_ptr/frame_ptr을 "그 ep로 이동"시키는 대신, 간단히 ep를 잘라서 소비된 부분 제거
                    # (메모리/속도 trade-off지만 구현이 단순)
                    if take2 < len(next_ep):
                        self.episodes[order[next_ep_ptr]] = next_ep[take2:]
                        break
                    else:
                        next_ep_ptr += 1

            if len(batch) < self.batch_size and self.drop_last:
                break

            yield batch

    def __len__(self) -> int:
        # upper bound: frames / batch_size
        if self.drop_last:
            return self._num_frames // self.batch_size
        return (self._num_frames + self.batch_size - 1) // self.batch_size