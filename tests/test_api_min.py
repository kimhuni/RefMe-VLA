# test_api_min.py
"""
한두 개 프레임을 직접 지정해서 Gemini/OpenAI 등 API를 호출하고
공통 JSONL 형식으로 저장하는 초경량 테스트.

예)
python test_api_min.py \
  --dataset_root /data/piper_push/lerobot_5hz \
  --derived_root /data/piper_push_derived \
  --dataset_name piper_push \
  --provider gemini --model gemini-2.5-pro \
  --task "Press the blue button on the table." \
  --prev_desc "No prior context." \
  --side /data/piper_push_derived/frames_1fps/chunk-000/episode_000000/side/frame_000000000.jpg \
  --wrist /data/.../wrist/frame_000000000.jpg \
  --timestamp_ms 0
"""
import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

from tests.providers import call_provider, ApiRequest, ApiResponse


def parse_args():
    p = argparse.ArgumentParser(description="Minimal API tests on 1-2 frames (multi-view)")
    p.add_argument("--dataset_root", type=Path, required=True)
    p.add_argument("--derived_root", type=Path, required=True)
    p.add_argument("--dataset_name", type=str, default="piper_push")

    # 직접 지정할 입력(필수)
    p.add_argument("--task", type=str, required=True)
    p.add_argument("--prev_desc", type=str, default="No prior context.")
    p.add_argument("--side", type=Path, required=True, help="side image path (table view)")
    p.add_argument("--wrist", type=Path, required=True, help="wrist image path")
    p.add_argument("--timestamp_ms", type=int, default=0)
    p.add_argument("--chunk", type=str, default="chunk-000")
    p.add_argument("--episode_id", type=str, default="episode_000000")

    # API 선택
    p.add_argument("--provider", type=str, choices=["gemini", "openai"], required=True)
    p.add_argument("--model", type=str, required=True)

    # 생성 하이퍼 params
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--max_tokens", type=int, default=128)

    # 출력 제어
    p.add_argument("--run_id", type=str, default=None, help="tests/<provider>/<run_id>.jsonl (기본: 날짜시간)")
    return p.parse_args()


def ensure_parents(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)


def main():
    args = parse_args()
    run_id = args.run_id or datetime.now().strftime("%Y%m%dT%H%M%S")
    # 출력 경로: /api_eval_B/tests/<provider>/<run_id>.jsonl
    out_dir = args.derived_root / "api_eval_B" / "tests" / args.provider
    ensure_parents(out_dir)
    out_f = out_dir / f"{run_id}.jsonl"

    # 요청 구성
    req = ApiRequest(
        task=args.task,
        prev_desc=args.prev_desc,
        images=[("side", str(args.side)), ("wrist", str(args.wrist))],
    )

    # 호출
    resp: ApiResponse = call_provider(
        provider=args.provider,
        model=args.model,
        req=req,
        temperature=args.temperature,
        top_p=args.top_p,
        max_output_tokens=args.max_tokens if (args.provider == "gemini" or args.provider == "openai") else None,
    )

    # 공통 레코드로 저장 (라인 1개)
    uid = f"{args.dataset_name}@{args.chunk.split('-')[-1]}-{args.episode_id.split('_')[-1]}-{args.timestamp_ms:012d}"
    record = {
        "uid": uid,
        "task": args.task,
        "chunk_id": args.chunk,
        "episode_id": args.episode_id,
        "timestamp_ms": args.timestamp_ms,
        "images": {"side": str(args.side), "wrist": str(args.wrist)},
        "prev_desc": args.prev_desc,
        "api_output": {
            "desc_1": resp.desc_1,
            "desc_2": resp.desc_2,
            "status": resp.status,
        },
        "meta": {
            "capture": {"fps": 1, "cameras": ["side", "wrist"]},
            "source": {"dataset": args.dataset_name, "root": str(args.dataset_root)},
            "generator": {
                "provider": args.provider,
                "model": args.model,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "max_tokens": args.max_tokens,
                "prompt_version": "B_eval_v3_test",
                "k": 1,
            },
        },
    }
    with open(out_f, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"[test_api_min] wrote 1 line -> {out_f}")
    print(json.dumps(record["api_output"], indent=2))


if __name__ == "__main__":
    main()