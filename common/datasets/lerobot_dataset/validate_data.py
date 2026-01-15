#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

"""
python common/datasets/lerobot_dataset/validate_data.py --root /data/ghkim/lerobot_data/wipe_the_window_ep150
"""


def iter_episode_parquets(dataset_root: Path):
    data_dir = dataset_root / "data"
    for chunk_dir in sorted(data_dir.glob("chunk-*")):
        if not chunk_dir.is_dir():
            continue
        for pq in sorted(chunk_dir.glob("episode_*.parquet")):
            yield pq


def to_action_array(x):
    """
    LeRobot parquet에서 action 컬럼은 보통 numpy array / list 형태.
    안전하게 np.ndarray shape (7,) 로 변환.
    """
    if x is None:
        return None
    if isinstance(x, np.ndarray):
        return x
    # list, tuple 등
    try:
        return np.asarray(x)
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=str,
        required=True,
        help="LeRobotDataset root path (contains data/, meta/, videos/)",
    )
    parser.add_argument("--xyz_abs_ge", type=int, default=1000000)
    parser.add_argument("--r_abs_gt", type=int, default=180001)
    parser.add_argument("--gripper_abs_gt", type=int, default=80000)
    parser.add_argument("--max_print", type=int, default=0,
                        help="0이면 제한 없이 출력, >0이면 최대 출력 개수 제한")
    args = parser.parse_args()

    root = Path(args.root)
    if not (root / "data").exists():
        raise FileNotFoundError(f"'data/' not found under: {root}")

    printed = 0

    for pq_path in iter_episode_parquets(root):
        # parquet 로드
        try:
            df = pd.read_parquet(pq_path)
        except Exception as e:
            print(f"[READ_FAIL] {pq_path} : {e}")
            continue

        if "action" not in df.columns:
            print(f"[NO_ACTION_COL] {pq_path}")
            continue

        # action을 한 줄씩 검사 (메모리 안전, 구현 단순)
        for row_i, act in enumerate(df["action"].values):
            act = to_action_array(act)
            if act is None:
                print(f"[BAD_ACTION] {pq_path} row={row_i} action=None/convert_fail")
                continue

            if act.shape[0] < 7:
                print(f"[BAD_ACTION] {pq_path} row={row_i} action_shape={act.shape} action={act}")
                continue

            # xyz: indices 0,1,2
            xyz = act[:3]
            bad_xyz = np.where(np.abs(xyz) >= args.xyz_abs_ge)[0]
            if bad_xyz.size > 0:
                # bad_xyz 는 0~2 기준이므로 실제 인덱스는 그대로
                vals = [(int(k), int(xyz[k])) for k in bad_xyz]
                print(f"[XYZ_OUTLIER] file={pq_path} row={row_i} hits={vals} full_action={act.tolist()}")
                printed += 1
                if args.max_print and printed >= args.max_print:
                    return

            # rxyz: indices 3,4,5
            r = act[3:6]
            bad_r = np.where(np.abs(r) > args.r_abs_gt)[0]
            if bad_r.size > 0:
                # bad_r 는 0~2 기준 -> 실제는 +3
                vals = [(int(k + 3), int(r[k])) for k in bad_r]
                print(f"[R_OUTLIER]  file={pq_path} row={row_i} hits={vals} full_action={act.tolist()}")
                printed += 1
                if args.max_print and printed >= args.max_print:
                    return

            # gripper: index 6
            g = act[6]
            if np.abs(g) > args.gripper_abs_gt:
                print(f"[GRIP_OUTLIER] file={pq_path} row={row_i} idx=6 val={int(g)} full_action={act.tolist()}")
                printed += 1
                if args.max_print and printed >= args.max_print:
                    return


if __name__ == "__main__":
    main()