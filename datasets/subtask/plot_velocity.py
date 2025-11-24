#!/usr/bin/env python
"""
python datasets/subtask/plot_velocity.py piper_mix_v01_ep5/data/chunk-000/episode_000000.parquet
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
    """
    df = pd.read_parquet(parquet_path)

    if state_col not in df.columns:
        raise KeyError(
            f"Column '{state_col}' not found in parquet. "
            f"Available columns: {list(df.columns)}"
        )

    # 각 row의 observation.state 가 리스트/ndarray 라고 가정
    states = np.stack(df[state_col].to_numpy())  # shape: (T, 7) 정도
    pos = states[:, :3]  # (x, y, z)만 사용
    z = pos[:, 2]  # (T,)

    # 위치 차이 → 속도 크기
    dp = np.diff(pos, axis=0)  # shape: (T-1, 3)
    dist = np.linalg.norm(dp, axis=1)  # 프레임 간 이동 거리

    # dt 계산: timestamp 있으면 그걸 사용, 없으면 freq_hz 기반
    if "timestamp" in df.columns:
        ts = df["timestamp"].to_numpy()
        dt = np.diff(ts)  # (T-1,)
        # dt 가 0 이거나 이상한 값이 있을 수 있으니 최소 epsilon 적용
        eps = 1e-6
        dt = np.clip(dt, eps, None)
        v = dist / dt
        t = ts[1:]  # 속도는 두 프레임 사이, 일단 뒤 프레임 시각에 맞추자
    else:
        dt = 1.0 / freq_hz
        v = dist / dt
        # 프레임 인덱스 기준 시간 (0, dt, 2dt, ...)
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
        "parquet_path",
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

    title = os.path.basename(args.parquet_path)
    plot_velocity(t, v, z, title=title)


if __name__ == "__main__":
    main()
