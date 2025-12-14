from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd

from refme_datasets.subtask.config import VelocityLabelConfig


@dataclass
class LowSpeedSegment:
    """
    v < tau_low 인 연속 구간을 frame 인덱스로 표현한 구조체.
    frame_start / frame_end 는 둘 다 inclusive (0 ~ num_frames-1 기준).
    """
    frame_start: int
    frame_end: int

    @property
    def length(self) -> int:
        return self.frame_end - self.frame_start + 1


def compute_velocity_from_df(
    df: pd.DataFrame,
    state_col: str,
    cfg: VelocityLabelConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    DataFrame에서 observation.state (또는 유사 컬럼)를 읽어
    end-effector (x, y, z) 기반 속도 시퀀스를 계산한다.

    Args:
        df: episode 단위 parquet를 읽은 DataFrame (길이 T)
        state_col: EE state가 들어있는 컬럼명 (예: "observation.state")
        cfg: VelocityLabelConfig (freq_hz 사용)

    Returns:
        t: shape (T-1,) 시간축 (초). 각 원소는 "두 프레임 사이"의 뒤쪽 프레임 시각.
        v: shape (T-1,) 속도 크기 (m/s 같은 단위)
    """
    if state_col not in df.columns:
        raise KeyError(
            f"Column '{state_col}' not found in dataframe. "
            f"Available columns: {list(df.columns)}"
        )

    # 각 row 의 state가 array-like 라고 가정 (x, y, z, rx, ry, rz, theta)
    states = np.stack(df[state_col].to_numpy())  # (T, D)
    pos = states[:, :3]  # (x, y, z)만 사용

    # 위치 차이 → 프레임 간 이동 거리
    dp = np.diff(pos, axis=0)             # (T-1, 3)
    dist = np.linalg.norm(dp, axis=1)     # (T-1,)

    # 시간 간격 dt 계산
    if "timestamp" in df.columns:
        ts = df["timestamp"].to_numpy()
        dt = np.diff(ts)                  # (T-1,)
        eps = 1e-6
        dt = np.clip(dt, eps, None)
        v = dist / dt
        t = ts[1:]
    else:
        dt = 1.0 / cfg.freq_hz
        v = dist / dt
        T = len(df)
        t = np.arange(1, T) * dt

    return t, v


def find_low_speed_segments(
    t: np.ndarray,
    v: np.ndarray,
    cfg: VelocityLabelConfig,
) -> List[LowSpeedSegment]:
    """
    속도 시퀀스에서 v < tau_low 인 연속 구간들을 찾아서 반환한다.

    여기서 v[i] 는 "frame i → frame i+1" 구간에 대한 속도라고 보고,
    해당 속도를 frame (i+1) 의 속도로 매핑한다.

    - 처음 cfg.ignore_first_sec 이전 구간은 무시한다.
    - 연속된 True 구간을 하나의 segment로 본다.
    - 너무 짧은 segment는 (cfg.min_segment_sec 이하) 노이즈로 버릴 수 있다.

    Returns:
        segments: LowSpeedSegment 리스트 (frame_start, frame_end inclusive)
    """
    assert t.shape == v.shape
    num_vel = len(v)

    # 처음 ignore_first_sec 이전 구간은 threshold에서 제외
    mask_time = t >= cfg.ignore_first_sec
    mask_low = (v < cfg.tau_low) & mask_time

    # low-speed frame 인덱스 (df 기준 frame index)
    # v[i] → frame (i+1)에 해당한다고 정의
    low_frames = np.where(mask_low)[0] + 1  # 1 ~ T-1 범위

    segments: List[LowSpeedSegment] = []
    if len(low_frames) == 0:
        return segments

    # 연속 구간(run)으로 묶기
    run_start = low_frames[0]
    prev = low_frames[0]

    for f in low_frames[1:]:
        if f == prev + 1:
            # 같은 run 계속
            prev = f
            continue
        # run 종료
        segments.append(LowSpeedSegment(frame_start=run_start, frame_end=prev))
        run_start = f
        prev = f
    # 마지막 run
    segments.append(LowSpeedSegment(frame_start=run_start, frame_end=prev))

    # 너무 짧은 segment는 제거 (길이 기준은 초 단위)
    if cfg.min_segment_sec is not None and cfg.min_segment_sec > 0:
        min_len_frames = int(round(cfg.min_segment_sec * cfg.freq_hz))
        segments = [seg for seg in segments if seg.length >= min_len_frames]

    # 최소 간격(min_gap_sec) 적용:
    # segment 사이의 간격이 너무 좁으면 하나로 합치거나 뒤의 것을 버리는 정책을 선택할 수 있다.
    # 여기서는 "앞 segment의 end와 뒤 segment의 start 사이의 시간이 너무 짧으면"
    # 두 segment를 하나로 합친다.
    if cfg.min_gap_sec is not None and cfg.min_gap_sec > 0:
        min_gap_frames = int(round(cfg.min_gap_sec * cfg.freq_hz))
        merged: List[LowSpeedSegment] = []
        for seg in segments:
            if not merged:
                merged.append(seg)
                continue
            prev_seg = merged[-1]
            gap = seg.frame_start - prev_seg.frame_end
            if gap < min_gap_frames:
                # gap 이 너무 짧으면 하나로 합치기 (앞 segment의 start 유지, end 를 확장)
                merged[-1] = LowSpeedSegment(
                    frame_start=prev_seg.frame_start,
                    frame_end=seg.frame_end,
                )
            else:
                merged.append(seg)
        segments = merged

    return segments


def assign_subtask_indices(
    num_frames: int,
    low_speed_segments: List[LowSpeedSegment],
    num_subtasks: int,
) -> np.ndarray:
    if num_subtasks <= 0:
        raise ValueError(f"num_subtasks must be > 0, got {num_subtasks}.")
    required = num_subtasks - 1
    if len(low_speed_segments) < required:
        raise RuntimeError(
            f"Not enough low-speed segments ({len(low_speed_segments)}) "
            f"for num_subtasks={num_subtasks}."
        )
    segments = sorted(low_speed_segments, key=lambda s: s.frame_start)
    boundaries = [seg.frame_start for seg in segments[:required]]
    subtask_idx = np.zeros(num_frames, dtype=np.int32)
    prev = 0
    for k, b in enumerate(boundaries):
        end = max(b - 1, prev)
        subtask_idx[prev:end+1] = k
        prev = b
    subtask_idx[prev:] = num_subtasks - 1
    return subtask_idx