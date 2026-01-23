"""
merge_helm_datasets.py

여러 Task 폴더에 흩어진 jsonl 파일들을 읽어 하나의 train/val 파일로 병합합니다.
기존 파일명에 'train'이 포함되면 train set으로, 'val'이 포함되면 val set으로 분류합니다.


python helm_datasets_video/merge_helm_video_data.py \
  --src_root "/data/ghkim/helm_data/helm_video_task_10/jsonl_v4" \
  --dst_root "/data/ghkim/helm_data/helm_video_task_10/merged"
"""

import argparse
import random
import glob
import os
from pathlib import Path
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_root", type=str, required=True, help="Task 폴더들이 있는 최상위 경로 (예: .../jsonl_v4)")
    parser.add_argument("--dst_root", type=str, required=True, help="병합된 파일이 저장될 경로")
    parser.add_argument("--seed", type=int, default=42, help="Shuffle seed")
    args = parser.parse_args()

    src_path = Path(args.src_root)
    dst_path = Path(args.dst_root)
    dst_path.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)

    # 데이터를 담을 리스트
    train_lines = []
    val_lines = []

    # src_root 하위의 모든 .jsonl 파일 검색 (재귀)
    # 예: src_root/task_A/detect/train-00000.jsonl
    print(f"[Info] Searching .jsonl files in {src_path} ...")
    all_files = sorted(src_path.rglob("*.jsonl"))

    if not all_files:
        print(f"[Error] No .jsonl files found in {src_path}")
        return

    print(f"[Info] Found {len(all_files)} files. Merging...")

    for file_path in tqdm(all_files):
        fname = file_path.name

        # 파일명에 따른 분류 (build_helm.py 생성 규칙 따름)
        if "train" in fname:
            target_list = train_lines
        elif "val" in fname:
            target_list = val_lines
        else:
            print(f"[Warning] Skipping ambiguous file: {file_path}")
            continue

        # 파일 읽어서 리스트에 추가
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        target_list.append(line)
        except Exception as e:
            print(f"[Error] Failed to read {file_path}: {e}")

    # 데이터 셔플 (순서 섞기)
    print(f"[Info] Shuffling data... (Train: {len(train_lines)}, Val: {len(val_lines)})")
    random.shuffle(train_lines)
    random.shuffle(val_lines)

    # 병합 파일 저장
    out_train = dst_path / "all_train.jsonl"
    out_val = dst_path / "all_val.jsonl"

    print(f"[Info] Writing to {out_train} ...")
    with open(out_train, "w", encoding="utf-8") as f:
        for line in train_lines:
            f.write(line + "\n")

    print(f"[Info] Writing to {out_val} ...")
    with open(out_val, "w", encoding="utf-8") as f:
        for line in val_lines:
            f.write(line + "\n")

    print(f"[Done] Merge Complete!")
    print(f"  - Train: {len(train_lines)} samples")
    print(f"  - Val  : {len(val_lines)} samples")


if __name__ == "__main__":
    main()