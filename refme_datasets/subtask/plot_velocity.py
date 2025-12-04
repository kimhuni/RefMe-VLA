#!/usr/bin/env python
"""
python datasets/subtask/plot_velocity.py piper_mix_v01_ep5/data/chunk-000/episode_000000.parquet

python refme_datasets/subtask/plot_velocity.py --parquet_path "/Users/ghkim/data/RefMe_data/press the blue button two times/lerobot/data/chunk-000/episode_000000.parquet" --freq-hz 30

"""


import argparse
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def compute_velocity_from_parquet(
    parquet_path: str,
    state_col: str = "observation.state",
    freq_hz: float = 5.0,
):
    """
    단일 episode parquet 파일에서 (x, y, z) 기반 속도 시퀀스를 계산한다.

    Returns:
        t: (T-1,) 시간축 (초 단위)
        v: (T-1,) 속도 값
        z: (T,) z 좌표 시퀀스
    """
    df = pd.read_parquet(parquet_path)

    if state_col not in df.columns:
        raise KeyError(
            f"Column '{state_col}' not found in parquet. "
            f"Available columns: {list(df.columns)}"
        )

    # 각 row의 observation.state 가 "길이 1짜리 배열 안에 7D 배열"인 형식 처리
    raw_states = df[state_col].to_numpy()
    first = raw_states[0]

    if isinstance(first, np.ndarray) and first.ndim == 1 and isinstance(first[0], np.ndarray):
        # 각 row: array([array([...7D...])])
        states = np.stack([s[0] for s in raw_states])  # (T, 7)
    elif isinstance(first, np.ndarray):
        # 이미 바로 1D 벡터인 경우
        states = np.stack(raw_states)                  # (T, D)
    else:
        # 최후 fallback
        states = np.stack([np.array(s) for s in raw_states])

    pos = states[:, :3]  # (x, y, z)만 사용
    z = pos[:, 2]        # (T,)

    # 위치 차이 → 속도 크기
    dp = np.diff(pos, axis=0)       # (T-1, 3)
    dist = np.linalg.norm(dp, axis=1)  # (T-1,)

    # dt 계산: timestamp 있으면 그걸 사용, 없으면 freq_hz 기반
    if "timestamp" in df.columns:
        ts = df["timestamp"].to_numpy()
        dt = np.diff(ts)  # (T-1,)
        eps = 1e-6
        dt = np.clip(dt, eps, None)
        v = dist / dt
        t = ts[1:]
    else:
        dt = 1.0 / freq_hz
        v = dist / dt
        T = len(df)
        t = np.arange(1, T) * dt

    return t, v, z


def plot_velocity(t, v, z, title: str = None):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    ax1.plot(t, v)
    ax1.set_ylabel("EE speed")
    ax1.grid(True, alpha=0.3)
    if title is not None:
        ax1.set_title(title)

    ax2.plot(t, z[1:])  # z must align with t (T-1)
    ax2.set_xlabel("Time [s]")
    ax2.set_ylabel("Z position")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="단일 lerobot episode parquet에서 end-effector 속도를 plot하는 스크립트"
    )
    parser.add_argument(
        "--parquet_path",
        type=str,
        help="episode_XXXXXX.parquet 파일 경로",
    )
    parser.add_argument(
        "--state-col",
        type=str,
        default="observation.state",
        help="EE state가 들어있는 컬럼명 (기본: observation.state)",
    )
    parser.add_argument(
        "--freq-hz",
        type=float,
        default=5.0,
        help="timestamp 없을 때 사용할 샘플링 주파수 (기본: 5.0 Hz)",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.parquet_path):
        raise FileNotFoundError(f"Parquet file not found: {args.parquet_path}")

    t, v, z = compute_velocity_from_parquet(
        args.parquet_path, state_col=args.state_col, freq_hz=args.freq_hz
    )

    print(v)

    title = os.path.basename(args.parquet_path)
    plot_velocity(t, v, z, title=title)


if __name__ == "__main__":
    main()
