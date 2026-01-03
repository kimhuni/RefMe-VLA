from __future__ import annotations

import argparse
import glob
import json
import os
import random
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

"""
# 전부 병합
python helm_datasets_v3/merge_helm_data.py \
  --jsonl_root /data/ghkim/helm_data/press_the_button_nolight/jsonl_v3 \
  --out_dir   /data/ghkim/helm_data/press_the_button_nolight/jsonl_v3/merged/press_button_in_order \
  --tasks "press_RGB","press_RBG","press_GBR","press_GRB","press_BRG","press_BGR" \
  --split_mode keep \
  --shard_size 0
  
python helm_datasets_v3/merge_helm_data.py \
  --jsonl_root /data/ghkim/helm_test/press_the_button_nolight/jsonl_v3 \
  --out_dir   /data/ghkim/helm_test/press_the_button_nolight/jsonl_v3/merged \
  --split_mode keep \
  --shard_size 0
  
# task별 비율 맞추기
python merge_helm_jsonl.py \
  --jsonl_root .../jsonl \
  --out_dir .../merged_weighted \
  --tasks press_blue_button_1,press_blue_button_2 \
  --weights press_blue_button_1=1,press_blue_button_2=0.5
"""

def iter_jsonl_files(task_dir: Path, split: str) -> List[Path]:
    """
    Find jsonl shards for a split under a task directory.

    Supports both layouts:
      - {task_dir}/{split}-00000.jsonl
      - {task_dir}/detect/{split}-00000.jsonl
      - {task_dir}/update/{split}-00000.jsonl
      - any deeper nesting under task_dir
    """
    files = sorted(task_dir.rglob(f"{split}-*.jsonl"))
    return files

def iter_rows(files: List[Path]) -> Iterable[dict]:
    for fp in files:
        with fp.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except Exception as e:
                    raise RuntimeError(f"JSON parse error: {fp} line {line_no}: {e}")

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def parse_csv(s: Optional[str]) -> Optional[List[str]]:
    if not s:
        return None
    xs = [x.strip() for x in s.split(",") if x.strip()]
    return xs or None

def parse_weights(s: Optional[str]) -> Dict[str, float]:
    """
    "taskA=1,taskB=0.5" -> {"taskA":1.0, "taskB":0.5}
    """
    w: Dict[str, float] = {}
    if not s:
        return w
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError(f"Bad weight spec: {part} (expected task_id=weight)")
        k, v = part.split("=", 1)
        w[k.strip()] = float(v.strip())
    return w

def maybe_sample_rows(rows: List[dict], weight: float, rng: random.Random) -> List[dict]:
    """
    weight=1.0: keep all
    weight<1.0: downsample by probability
    weight>1.0: upsample by duplication (stochastic)
    """
    if weight == 1.0:
        return rows

    out: List[dict] = []
    if weight < 1.0:
        p = max(0.0, min(1.0, weight))
        for r in rows:
            if rng.random() < p:
                out.append(r)
        return out

    # weight > 1.0
    k = int(weight)
    frac = weight - k
    for r in rows:
        for _ in range(k):
            out.append(r)
        if rng.random() < frac:
            out.append(r)
    return out

def get_task_dirs(jsonl_root: Path, tasks: Optional[List[str]] = None) -> List[Path]:
    if tasks is None:
        # 모든 하위 폴더를 task로 취급
        return sorted([p for p in jsonl_root.iterdir() if p.is_dir()])
    out = []
    for t in tasks:
        p = jsonl_root / t
        if not p.exists():
            raise FileNotFoundError(f"Task dir not found: {p}")
        out.append(p)
    return out

def dedup_by_uid(rows: List[dict]) -> List[dict]:
    seen: Set[str] = set()
    out: List[dict] = []
    for r in rows:
        uid = str(r.get("uid", ""))
        if not uid:
            # uid가 없으면 그냥 포함 (원하면 여기서 에러로 바꿔도 됨)
            out.append(r)
            continue
        if uid in seen:
            continue
        seen.add(uid)
        out.append(r)
    return out

def write_jsonl(path: Path, rows: List[dict]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def shard_write(path_prefix: Path, rows: List[dict], shard_size: int) -> List[Path]:
    """
    path_prefix 예: out_dir/"all_train"
    -> all_train-00000.jsonl, all_train-00001.jsonl ...
    """
    ensure_dir(path_prefix.parent)
    out_files: List[Path] = []
    if shard_size <= 0:
        p = path_prefix.with_suffix(".jsonl")
        write_jsonl(p, rows)
        return [p]

    for i in range(0, len(rows), shard_size):
        shard = rows[i:i+shard_size]
        p = path_prefix.parent / f"{path_prefix.name}-{i//shard_size:05d}.jsonl"
        write_jsonl(p, shard)
        out_files.append(p)
    return out_files

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl_root", type=str, required=True,
                    help=".../out_root/jsonl (task별 폴더가 있는 위치)")
    ap.add_argument("--out_dir", type=str, required=True,
                    help="병합 결과 저장 폴더")
    ap.add_argument("--tasks", type=str, default=None,
                    help="병합할 task_id 목록(콤마). 비우면 jsonl_root 아래 전부 병합")
    ap.add_argument("--weights", type=str, default=None,
                    help="task별 weight. 예: taskA=1,taskB=0.5 (다운/업샘플)")
    ap.add_argument("--seed", type=int, default=1234)

    ap.add_argument("--split_mode", type=str, default="keep",
                    choices=["keep", "train_only", "rebuild_from_all"],
                    help=(
                        "keep: task별로 만들어진 train/val을 그대로 합침(기본)\n"
                        "train_only: val 무시하고 train만 합침\n"
                        "rebuild_from_all: 모든 row를 모은 뒤 episode 단위로 다시 train/val split (아래 val_ratio 사용)"
                    ))
    ap.add_argument("--val_ratio", type=float, default=0.1,
                    help="split_mode=rebuild_from_all에서만 사용")

    ap.add_argument("--dedup_uid", type=int, default=1,
                    help="1이면 uid 기준 중복 제거")
    ap.add_argument("--shuffle", type=int, default=1)
    ap.add_argument("--shard_size", type=int, default=0,
                    help="0이면 단일 파일로 저장, >0이면 shard로 나눔")

    args = ap.parse_args()

    jsonl_root = Path(args.jsonl_root)
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    tasks = parse_csv(args.tasks)
    weights = parse_weights(args.weights)
    rng = random.Random(args.seed)

    task_dirs = get_task_dirs(jsonl_root, tasks)

    def load_split(split: str) -> List[dict]:
        rows_all: List[dict] = []
        for td in task_dirs:
            task_id = td.name
            files = iter_jsonl_files(td, split)
            if not files:
                # e.g., task_dir may have only train or only val
                # or split shards might be stored elsewhere.
                # Keep silent behavior but make it debuggable when needed.
                # (uncomment next line if you want verbose logs)
                # print(f"[warn] no {split} jsonl under: {td}")
                continue
            rows = list(iter_rows(files))
            w = weights.get(task_id, 1.0)
            rows = maybe_sample_rows(rows, w, rng)
            rows_all.extend(rows)
        return rows_all

    if args.split_mode == "keep":
        train_rows = load_split("train")
        val_rows = load_split("val")

    elif args.split_mode == "train_only":
        train_rows = load_split("train")
        val_rows = []

    else:  # rebuild_from_all
        all_rows = load_split("train") + load_split("val")

        # episode 단위 split (chunk, episode) 기준
        ep_keys: List[Tuple[str, str]] = []
        for r in all_rows:
            ep_keys.append((str(r.get("chunk", "")), str(r.get("episode", ""))))
        uniq = list(dict.fromkeys(ep_keys))
        rng.shuffle(uniq)
        n_val = int(len(uniq) * float(args.val_ratio))
        val_set = set(uniq[:n_val])

        train_rows, val_rows = [], []
        for r in all_rows:
            key = (str(r.get("chunk", "")), str(r.get("episode", "")))
            (val_rows if key in val_set else train_rows).append(r)

    if args.dedup_uid:
        train_rows = dedup_by_uid(train_rows)
        val_rows = dedup_by_uid(val_rows)

    if args.shuffle:
        rng.shuffle(train_rows)
        rng.shuffle(val_rows)

    # 저장
    train_files = shard_write(out_dir / "all_train", train_rows, args.shard_size)
    val_files = shard_write(out_dir / "all_val", val_rows, args.shard_size)

    meta = {
        "jsonl_root": str(jsonl_root),
        "tasks": [p.name for p in task_dirs],
        "weights": weights,
        "split_mode": args.split_mode,
        "val_ratio": args.val_ratio if args.split_mode == "rebuild_from_all" else None,
        "dedup_uid": bool(args.dedup_uid),
        "shuffle": bool(args.shuffle),
        "counts": {
            "train_rows": len(train_rows),
            "val_rows": len(val_rows),
            "train_files": [str(p) for p in train_files],
            "val_files": [str(p) for p in val_files],
        },
        "note": "Images are referenced by path stored in each row['images'] (no copying).",
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Saved train rows={len(train_rows)} -> {out_dir}")
    print(f"Saved val rows={len(val_rows)} -> {out_dir}")

if __name__ == "__main__":
    main()