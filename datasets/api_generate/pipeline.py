# pipeline.py
import argparse
import logging
import json
from pathlib import Path

from datasets.api_generate.config import Paths, GenConfig
from datasets.api_generate.io_utils import (
    ensure_frames_for_episode,
    episode_video_paths,
    iter_aligned_pairs,
    iter_chunks,
    list_episodes_in_chunk,
    load_episodes_meta,
    reset_file,
    write_jsonl,
)
from datasets.api_generate.api_client import ApiClient, ApiRequest  # ← API 호출


def run_pipeline(paths: Paths, cfg: GenConfig, dataset_name: str, fps: int = 1, revise: bool = False, target_episodes: set = None) -> None:
    # 출력 디렉토리 (프롬프트별 네임스페이스)
    out_root = paths.out_root(cfg.prompt_id)
    shards_dir = out_root / "shards"
    stats_dir = out_root / "stats"
    index_file = out_root / "dataset.index"
    failures_file = out_root / "failures.jsonl"

    out_root.mkdir(parents=True, exist_ok=True)
    shards_dir.mkdir(parents=True, exist_ok=True)
    stats_dir.mkdir(parents=True, exist_ok=True)
    if not revise:
        reset_file(index_file)
        reset_file(failures_file)

    # 메타 로드 (task 등)
    episodes_meta = load_episodes_meta(paths.episodes_meta)

    # ApiClient 준비
    api = ApiClient(
        provider=cfg.provider,
        model=cfg.model,
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        max_tokens=cfg.max_tokens,
        seed=cfg.seed,
        prompt_id=cfg.prompt_id,
    )

    total_chunks = 0
    total_episodes = 0
    total_pairs = 0
    total_records = 0

    for chunk_name, chunk_dir in iter_chunks(paths.videos):
        total_chunks += 1
        shard_wrote_any = False
        shard_path = shards_dir / f"{chunk_name}.jsonl"
        if revise:
            # write new lines to a temporary file first
            new_lines_path = shards_dir / f"{chunk_name}.jsonl.revise_new"
            reset_file(new_lines_path)
            shard_writer_path = new_lines_path
        else:
            reset_file(shard_path)
            shard_writer_path = shard_path

        episodes = list_episodes_in_chunk(chunk_dir)
        if revise and target_episodes: # override
            episodes = [e for e in episodes if e in target_episodes]
        total_episodes += len(episodes)
        logging.info("[chunk %s] episodes=%d", chunk_name, len(episodes))

        for epi_idx in episodes:
            # mp4 경로
            table_mp4, wrist_mp4 = episode_video_paths(chunk_dir, epi_idx)

            # 프레임 추출 위치 (기존 구조 준수)
            out_side = paths.frames_root / chunk_name / f"episode_{epi_idx:06d}" / "side"
            out_wrist = paths.frames_root / chunk_name / f"episode_{epi_idx:06d}" / "wrist"

            # 1) 프레임 추출(이미 있으면 스킵)
            ensure_frames_for_episode(table_mp4, wrist_mp4, out_side, out_wrist, fps=fps)

            # 2) 멀티뷰 정합 프레임
            pairs = list(iter_aligned_pairs(out_side, out_wrist))
            total_pairs += len(pairs)
            logging.info("[episode %06d] aligned_pairs=%d", epi_idx, len(pairs))
            if not pairs:
                logging.warning("[episode %06d] no aligned frames (side/wrist)", epi_idx)
                continue

            # 3) task/prev_desc
            meta = episodes_meta.get(epi_idx, {})
            task = meta.get("task") or meta.get("tasks") or "unknown task"
            prev_desc = "No prior context."
            prev_status = "NOT_DONE"

            # 4) 프레임 단위 API 호출 → JSONL 저장
            for side_img, wrist_img, ts in pairs:
                uid = f"{dataset_name}@{chunk_name.split('-')[-1]}-{epi_idx:06d}-{ts:012d}"
                try:
                    req = ApiRequest(
                        task=task,
                        prev_desc=prev_desc,
                        prev_status=prev_status,
                        images=[("side", str(side_img)), ("wrist", str(wrist_img))],
                    )
                    resp = api.call(req)
                    if resp.status not in {"DONE", "NOT_DONE", "PARTIALLY_DONE"}:
                        raise ValueError(f"invalid status: {resp.status}")

                    # Sticky DONE guard
                    if prev_status == "DONE":
                        resp.status = "DONE"
                        # 선택: desc_2에 유지 근거 한 줄 고정
                        if not resp.desc_2.strip():
                            resp.desc_2 = "Previous frame DONE; task remains completed with no reversal visible."
                    print("current episode : ", epi_idx)

                    record = {
                        "uid": uid,
                        "task": task,
                        "chunk_id": chunk_name,
                        "episode_id": f"episode_{epi_idx:06d}",
                        "timestamp_ms": ts,
                        "images": {
                            "side": str(side_img),
                            "wrist": str(wrist_img),
                        },
                        "prev_desc": prev_desc,
                        "prev_status": prev_status,
                        "api_output": {
                            "desc_1": resp.desc,
                            "desc_2": resp.status_reasoning,
                            "status": resp.status,
                        },
                        "meta": {
                            "capture": {"fps": fps, "cameras": ["side", "wrist"]},
                            "source": {"dataset": dataset_name, "root": str(paths.dataset_root)},
                            "generator": {
                                "provider": cfg.provider,
                                "model": cfg.model,
                                "temperature": cfg.temperature,
                                "top_p": cfg.top_p,
                                "max_tokens": cfg.max_tokens,
                                "seed": cfg.seed,
                                "prompt_id": cfg.prompt_id,
                                "k": 1,
                            },
                        },
                    }
                    write_jsonl(shard_writer_path, record)
                    shard_wrote_any = True
                    total_records += 1

                    # prev_desc 갱신 (160자 캡)
                    # prev_desc = (resp.desc_2 + ": " + resp.status).strip()
                    prev_desc = resp.status_reasoning
                    prev_status = resp.status
                    if len(prev_desc) > cfg.prev_desc_max_chars:
                        prev_desc = prev_desc[: cfg.prev_desc_max_chars - 3] + "..."


                except Exception as e:
                    # 실패 로그
                    write_jsonl(failures_file, {
                        "uid": uid,
                        "chunk": chunk_name,
                        "episode": epi_idx,
                        "timestamp_ms": ts,
                        "reason": str(e),
                    })
                    # prev_desc는 유지

        if revise:
            final_tmp = shards_dir / f"{chunk_name}.jsonl.tmp"
            with open(final_tmp, "w", encoding="utf-8") as w:
                # 5-a) keep old lines for non-target episodes
                if shard_path.exists():
                    with open(shard_path, "r", encoding="utf-8") as r:
                        for line in r:
                            try:
                                o = json.loads(line)
                                epi_id = int(str(o.get("episode_id","episode_000000")).split("_")[-1])
                                if target_episodes and epi_id in target_episodes:
                                    continue
                                w.write(line)
                            except Exception:
                                # if malformed, keep as-is
                                w.write(line)
                # 5-b) append new lines
                if (shards_dir / f"{chunk_name}.jsonl.revise_new").exists():
                    with open(shards_dir / f"{chunk_name}.jsonl.revise_new", "r", encoding="utf-8") as nr:
                        for line in nr:
                            w.write(line)
            # atomically replace
            import os
            os.replace(final_tmp, shard_path)
            # cleanup new-lines temp
            try:
                (shards_dir / f"{chunk_name}.jsonl.revise_new").unlink()
            except Exception:
                pass

        # 샤드 인덱스 기록
        if shard_wrote_any:
            write_jsonl(index_file, {"path": str(shard_path), "chunk": chunk_name})

    logging.info("[summary] chunks=%d episodes=%d aligned_pairs=%d records=%d",
                 total_chunks, total_episodes, total_pairs, total_records)

def parse_episode_set(spec: str):
    if not spec:
        return None
    s = spec.strip()
    if not s:
        return None
    items = set()
    for tok in s.split(","):
        tok = tok.strip()
        if tok == "":
            continue
        # only simple comma-separated integers for now
        try:
            items.add(int(tok))
        except ValueError:
            raise ValueError(f"Invalid episode index: {tok}")
    return items


def parse_args():
    p = argparse.ArgumentParser(description="Generate API-eval JSONL (B-step) with multi-view images")
    p.add_argument("--dataset_root", type=Path, required=True, help="Path to LeRobot dataset root (e.g., /data/.../lerobot_5hz)")
    p.add_argument("--derived_root", type=Path, required=True, help="Path to write derived outputs (e.g., /data/..._derived)")
    p.add_argument("--dataset_name", type=str, required=True, help="Dataset name tag for uid")
    p.add_argument("--fps", type=int, default=1)
    p.add_argument("--provider", type=str, default="gemini", choices=["gemini", "openai", "custom"])
    p.add_argument("--model", type=str, default="gemini-1.5-pro")
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--max_tokens", type=int, default=384)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--prompt_id", type=str, default="B_eval_v3", help="Choose from B_eval_v3, B_eval_compact, B_eval_strict_json")
    p.add_argument("--revise", action="store_true", default=False,help="If set, only re-generate selected episodes.")
    p.add_argument("--episodes", type=str, default="", help="Comma-separated list of episode indices to regenerate, e.g., '0,1,3,5'")
    p.add_argument("--verbose", action="store_true", default=False, help="Enable verbose logging")
    return p.parse_args()


def main():
    args = parse_args()
    if args.verbose:
        logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

    paths = Paths(dataset_root=args.dataset_root, derived_root=args.derived_root)

    # 입력 경로 가드
    if not paths.dataset_root.exists():
        logging.error("Dataset root path does not exist: %s", paths.dataset_root)
        return
    if not paths.videos.exists():
        logging.error("Videos path does not exist: %s", paths.videos)
        return

    # GenConfig 구성
    cfg = GenConfig(
        fps=args.fps,
        provider=args.provider,
        model=args.model,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        seed=args.seed,
        prompt_id=args.prompt_id,
    )

    target_episodes = parse_episode_set(args.episodes) if args.revise else None
    run_pipeline(paths, cfg, dataset_name=args.dataset_name, fps=args.fps, revise=args.revise, target_episodes=target_episodes)


if __name__ == "__main__":
    main()
