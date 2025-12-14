#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sticky DONE post-processor for shard JSONL.
- 에피소드 단위로 첫 DONE 이후의 모든 프레임에 대해 status를 DONE으로 고정
- desc_2는 기본 메시지로 덮어쓰기(기본: 항상 덮어씀)

사용 예:
# (1) 단일 파일: shards/ 밑 chunk-000.jsonl 을 읽어서 shards_DONE/chunk-000.jsonl 로 저장
python done_desc.py \
  --in_path /data/ghkim/piper_press_the_blue_button_ep60/gpt-5-mini/eval_final/shards/chunk-000.jsonl

# (2) 디렉터리 일괄 처리: shards/ 안의 *.jsonl 모두 처리 → shards_DONE/ 로 저장
python done_desc.py \
  --in_dir  /data/ghkim/piper_press_the_blue_button_ep60/gpt-5-mini/eval_final/shards

# (3) 재귀 처리(rglob 사용): 하위 폴더까지 *.jsonl 모두 처리
python done_desc.py \
  --in_dir  /data/.../eval_final \
  --recursive

옵션:
--overwrite_desc  : 첫 DONE 이후 구간에서 desc_2를 항상 메시지로 덮어쓰기(기본: 항상 덮어씀)
--message "..."   : desc_2에 채울 메시지( overwrite_desc=True 이면 항상 적용 )
"""

import argparse
import json
from pathlib import Path
from collections import defaultdict

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_path", type=Path, required=False, help="입력 shard JSONL 경로 (e.g., .../shards/chunk-000.jsonl)")
    ap.add_argument("--out_path", type=Path, default=None, help="출력 파일 경로(지정 없으면 shards_DONE/chunk-XXX.jsonl)")
    ap.add_argument("--out_dir", type=Path, default=None, help="출력 디렉터리(지정 시 그 아래 동일 파일명으로 저장)")
    ap.add_argument("--overwrite_desc", action="store_true", default=True, help="첫 DONE 이후 desc_2를 항상 메시지로 덮어쓰기(기본: True)")
    ap.add_argument("--message", type=str, default="Previous frame DONE; task remains completed with no visible change.",
                    help="desc_2에 채울 메시지( overwrite_desc=True 이면 항상 적용 )")
    ap.add_argument("--in_dir", type=Path, default=None, help="입력 디렉터리(.jsonl 일괄 처리)")
    ap.add_argument("--recursive", action="store_true", help="--in_dir 사용 시 하위 폴더까지 재귀 처리")
    return ap.parse_args()

def determine_out_path(in_path: Path, out_dir: Path =None, out_path: Path = None) -> Path:
    if out_path:
        return out_path
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir / in_path.name
    # 기본: 입력폴더 아래 shards_DONE/
    default_dir = in_path.parent.parent / "shards_DONE"
    default_dir.mkdir(parents=True, exist_ok=True)
    return default_dir / in_path.name

def process_one_file(in_path: Path, out_dir: Path = None, out_path: Path = None, overwrite_desc: bool = True, message: str = None):
    out_path_resolved = determine_out_path(in_path, out_dir, out_path)
    lines = load_lines(in_path)
    buckets = group_by_episode(lines)

    original_order_keys = []
    seen = set()
    for item in lines:
        if isinstance(item, dict):
            k = item.get("episode_id") if isinstance(item.get("episode_id"), str) else "__NO_EPISODE__"
        else:
            k = "__RAW__"
        if k not in seen:
            seen.add(k)
            original_order_keys.append(k)

    for epi_id, recs in list(buckets.items()):
        if epi_id in ("__RAW__", "__NO_EPISODE__"):
            continue
        recs_sorted = sort_episode_records([r for r in recs if isinstance(r, dict)])
        recs_fixed = apply_sticky_done(recs_sorted, overwrite_desc, message)
        raws = [r for r in recs if not isinstance(r, dict)]
        buckets[epi_id] = recs_fixed + raws

    write_lines(out_path_resolved, buckets, original_order_keys)
    print(f"[done_desc] wrote: {out_path_resolved}")

def load_lines(in_path: Path):
    lines = []
    with in_path.open("r", encoding="utf-8") as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
                lines.append(obj)
            except Exception:
                # 깨진 라인은 그대로 보존
                lines.append(raw)
    return lines

def group_by_episode(lines):
    # episode_id(str) -> [records or raw strings]
    buckets = defaultdict(list)
    for item in lines:
        if isinstance(item, dict):
            epi = item.get("episode_id")
            # episode_id가 없으면 기타 버킷
            key = epi if isinstance(epi, str) else "__NO_EPISODE__"
            buckets[key].append(item)
        else:
            buckets["__RAW__"].append(item)
    return buckets

def sort_episode_records(recs):
    # timestamp_ms 기준 정렬; 없는 경우 원래 순서 유지
    try:
        return sorted(recs, key=lambda o: o.get("timestamp_ms", -1))
    except Exception:
        return recs

def apply_sticky_done(recs, overwrite_desc: bool, message: str):
    """
    첫 DONE이 등장한 시점(i)부터 이후 모든 j>=i에 대해:
    - api_output.status = "DONE"
    - desc_2: overwrite_desc=True면 무조건 message로, False면 비어있을 때만 message로 채움
    """
    # DONE 최초 인덱스 탐색
    first_done_idx = None
    for i, r in enumerate(recs):
        try:
            st = r.get("api_output", {}).get("status", "")
            if st == "DONE":
                first_done_idx = i
                break
        except Exception:
            pass

    if first_done_idx is None:
        return recs  # 변경 없음

    for j in range(first_done_idx, len(recs)):
        r = recs[j]
        try:
            ao = r.setdefault("api_output", {})
            ao["status"] = "DONE"
            if overwrite_desc:
                ao["desc_2"] = message
            else:
                d2 = ao.get("desc_2", "")
                if not isinstance(d2, str) or not d2.strip():
                    ao["desc_2"] = message
        except Exception:
            # 레코드 구조가 이상하면 건너뜀
            continue
    return recs

def write_lines(out_path: Path, buckets, original_order_keys):
    with out_path.open("w", encoding="utf-8") as w:
        for key in original_order_keys:
            items = buckets.get(key, [])
            for it in items:
                if isinstance(it, dict):
                    w.write(json.dumps(it, ensure_ascii=False) + "\n")
                else:
                    w.write(str(it) + "\n")

def main():
    args = parse_args()
    if (args.in_path is None) and (args.in_dir is None):
        raise SystemExit("Please provide --in_path FILE or --in_dir DIR")
    if (args.in_path is not None) and (args.in_dir is not None):
        raise SystemExit("Use either --in_path or --in_dir (not both)")

    if args.in_dir:
        root = args.in_dir
        if not root.exists():
            raise SystemExit(f"--in_dir not found: {root}")
        # 기본 out_dir: 입력 디렉터리의 부모에 shards_DONE/ 생성
        default_dir = root.parent / "shards_DONE"
        default_dir.mkdir(parents=True, exist_ok=True)
        pattern_iter = root.rglob("*.jsonl") if args.recursive else root.glob("*.jsonl")
        count = 0
        for fp in sorted(pattern_iter):
            # 각 파일은 같은 파일명으로 shards_DONE/ 밑에 저장
            process_one_file(fp, out_dir=default_dir, out_path=None,
                             overwrite_desc=args.overwrite_desc, message=args.message)
            count += 1
        print(f"[done_desc] processed {count} files under {root} -> {default_dir}")
        return

    # 단일 파일 모드
    in_path = args.in_path
    out_path = determine_out_path(in_path, args.out_dir, args.out_path)
    process_one_file(in_path, out_dir=args.out_dir, out_path=out_path,
                     overwrite_desc=args.overwrite_desc, message=args.message)

if __name__ == "__main__":
    main()