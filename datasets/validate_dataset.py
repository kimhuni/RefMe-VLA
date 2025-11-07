#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Usage examples:
python validate_dataset.py --path /data/ghkim/piper_press_the_blue_button_ep60/gpt-5-mini/eval_final --stats --out /tmp/validate_eval.tsv
python validate_dataset.py --path /path/to/eval_final --with-processor Qwen/Qwen2.5-VL-7B-Instruct --limit 64 --out /tmp/validate_eval.tsv
python datasets/validate_dataset.py --path /data/ghkim/piper_press_the_blue_button_ep60/gpt-5-mini/eval_final --include-root --stats --out /tmp/validate_eval.tsv
"""



import os, sys, json, glob, argparse
from typing import List, Dict, Any, Tuple, Optional
from PIL import Image
CHECK_UID = False
CHECK_EP_ORDER = False

# (선택) processor 검사 옵션
try:
    from transformers import AutoProcessor
    _HF_OK = True
except Exception:
    _HF_OK = False


CAM_FALLBACK_ORDER = ["side", "wrist"]

SEEN_UIDS = set()
EP_LAST_TS = {}
STRICT_IMAGES = False


def expand_jsonl_paths(path: str) -> List[str]:
    if os.path.isfile(path) and path.endswith(".jsonl"):
        return [path]
    if os.path.isdir(path):
        # If a 'shards' subdir exists, prefer ONLY files inside it to avoid duplicates
        shards_dir = os.path.join(path, "shards")
        use_dir = shards_dir if os.path.isdir(shards_dir) else path
        files = glob.glob(os.path.join(use_dir, "**", "*.jsonl"), recursive=True)
        files.sort()
        return files
    raise ValueError(f"Path {path} is neither .jsonl nor a directory: {path}")

def load_jsonl_lines(path: str) -> List[Tuple[int, Dict[str, Any]]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append((i, json.loads(line)))
            except Exception as e:
                rows.append((i, {"__parse_error__": str(e), "__raw__": line}))
    return rows

def row_identifier(row: Dict[str, Any]) -> str:
    if isinstance(row, dict):
        if "uid" in row and row["uid"]:
            return str(row["uid"])
        parts = []
        for k in ("chunk_id", "episode_id", "timestamp_ms"):
            v = row.get(k)
            if v is not None:
                parts.append(str(v))
        if parts:
            return "|".join(parts)
    return "unknown"

def validate_labels(row: Dict[str, Any]) -> List[str]:
    issues = []
    valid_statuses = {"DONE","NOT_DONE","UNCERTAIN"}
    def check_desc(desc_key, desc_val):
        if not isinstance(desc_val, str) or not desc_val.strip():
            issues.append(f"label_{desc_key}_empty")
        elif len(desc_val.strip()) > 512:
            issues.append(f"label_{desc_key}_too_long")

    if "api_output" in row and isinstance(row["api_output"], dict):
        ao = row["api_output"]
        for k in ("desc_1", "desc_2", "status"):
            if k not in ao:
                issues.append(f"api_output_missing_{k}")
        status = ao.get("status")
        if status not in valid_statuses:
            issues.append("label_status_invalid")
        check_desc("desc1", ao.get("desc_1", ""))
        check_desc("desc2", ao.get("desc_2", ""))
    elif "target_text" in row:
        try:
            parsed = json.loads(row["target_text"])
            if not isinstance(parsed, dict):
                issues.append("label_target_text_not_dict")
            else:
                status = parsed.get("status")
                if status not in valid_statuses:
                    issues.append("label_status_invalid")
                check_desc("desc1", parsed.get("desc_1", ""))
                check_desc("desc2", parsed.get("desc_2", ""))
        except Exception:
            issues.append("label_target_text_not_json")
    return issues

def validate_meta_and_images(row: Dict[str, Any]) -> List[str]:
    issues = []
    meta = row.get("meta", {})
    cap = meta.get("capture", {}) if isinstance(meta, dict) else {}
    cameras = None
    if "cameras" in cap:
        if not isinstance(cap["cameras"], list):
            issues.append("meta_cameras_not_list")
        else:
            cameras = cap["cameras"]
    images = row.get("images")
    if isinstance(images, dict):
        if not (1 <= len(images) <= 8):
            issues.append("images_count_out_of_bounds")
        for k,v in images.items():
            if not isinstance(v, str):
                issues.append(f"images_value_not_str[{k}]")
        if cameras is not None:
            if STRICT_IMAGES:
                if set(images.keys()) != set(cameras):
                    issues.append("images_keys_mismatch_strict")
            if len(images) != len(cameras):
                issues.append("images_count_mismatch")
    if "timestamp_ms" in row:
        ts = row["timestamp_ms"]
        try:
            ts_int = int(ts)
            if ts_int < 0:
                issues.append("timestamp_invalid")
        except Exception:
            issues.append("timestamp_invalid")
    return issues

# UID duplicate diagnostics
FIRST_SEEN_UID = {}
CURRENT_FILE = ""

def validate_uid_and_dups(row: Dict[str, Any], row_line: Optional[int]=None) -> List[str]:
    issues = []
    if not (CHECK_UID or CHECK_EP_ORDER):
        return issues
    if CHECK_UID:
        uid = row.get("uid")
        if uid is not None:
            if uid in SEEN_UIDS:
                orig = FIRST_SEEN_UID.get(uid)
                if orig:
                    issues.append(f"duplicate_uid(first_seen={os.path.basename(orig[0])}:{orig[1]})")
                else:
                    issues.append("duplicate_uid")
            else:
                SEEN_UIDS.add(uid)
                if row_line is not None:
                    FIRST_SEEN_UID[uid] = (CURRENT_FILE, row_line)
    if CHECK_EP_ORDER:
        ep_key = (row.get("chunk_id"), row.get("episode_id"))
        ts = row.get("timestamp_ms")
        if ep_key != (None, None) and ts is not None:
            try:
                ts_val = int(ts)
                last_ts = EP_LAST_TS.get(ep_key)
                if last_ts is not None and ts_val < last_ts:
                    issues.append("timestamp_non_monotonic")
                EP_LAST_TS[ep_key] = ts_val
            except Exception:
                # timestamp invalid handled elsewhere
                pass
    return issues

def check_required_keys(row: Dict[str, Any]) -> List[str]:
    issues = []
    # 이미지 키
    has_multi = isinstance(row.get("images"), dict)
    has_single = "image_path" in row
    if not (has_multi or has_single):
        issues.append("missing_image_field(images|image_path)")
    # 텍스트 키
    if "task" not in row:
        issues.append("missing_task")
    if ("prev_desc" not in row) and ("prev" not in row):
        issues.append("missing_prev_or_prev_desc")
    if "prev_status" not in row:
        issues.append("missing_prev_status")
    # 라벨 키
    has_target = "target_text" in row
    has_api = isinstance(row.get("api_output"), dict)
    if not (has_target or has_api):
        issues.append("missing_label(target_text|api_output)")
    return issues

def ordered_cameras(row: Dict[str, Any]) -> List[str]:
    cams = None
    meta = row.get("meta", {})
    cap = meta.get("capture", {}) if isinstance(meta, dict) else {}
    if isinstance(cap.get("cameras"), list) and cap["cameras"]:
        cams = cap["cameras"]
    elif isinstance(row.get("images"), dict):
        present = list(row["images"].keys())
        ordered = [c for c in CAM_FALLBACK_ORDER if c in present]
        ordered += sorted([c for c in present if c not in CAM_FALLBACK_ORDER])
        cams = ordered
    return cams or []

def check_images_exist_and_open(row: Dict[str, Any]) -> List[str]:
    issues = []
    if isinstance(row.get("images"), dict):
        cams = ordered_cameras(row)
        if not cams:
            issues.append("images_no_cameras_order")
        else:
            for cam in cams:
                p = row["images"].get(cam)
                if not isinstance(p, str):
                    issues.append(f"image_path_not_str[{cam}]")
                    continue
                if not os.path.isfile(p):
                    issues.append(f"image_missing[{cam}]={p}")
                    continue
                try:
                    Image.open(p).convert("RGB").close()
                except Exception as e:
                    issues.append(f"image_open_fail[{cam}]={type(e).__name__}")
    elif "image_path" in row:
        p = row["image_path"]
        if not os.path.isfile(p):
            issues.append(f"image_missing[path]={p}")
        else:
            try:
                Image.open(p).convert("RGB").close()
            except Exception as e:
                issues.append(f"image_open_fail[path]={type(e).__name__}")
    return issues

def build_prompt_min(row: Dict[str, Any]) -> str:
    task = row.get("task", "")
    prev = row.get("prev_desc", row.get("prev", ""))
    prev_status = row.get("prev_status", "UNCERTAIN")
    return (
        "You are an image-analysis expert for robot manipulation.\n"
        "IMAGES: [SIDE]=global scene view; [WRIST]=close-up wrist camera.\n"
        f"TASK: {task}\n"
        f"PREV_DESC: {prev}\n"
        f"PREV_STATUS: {prev_status}\n"
        "Describe what is visibly happening now (desc_1) and the visible evidence for completion (desc_2).\n"
        "Then decide the status: DONE / NOT_DONE / UNCERTAIN.\n"
        "Output JSON: {\"desc_1\":\"...\",\"desc_2\":\"...\",\"status\":\"...\"}"
    )

def normalize_grid_thw(val):
    # processor가 반환한 grid를 ( [ [t,h,w], ... ] )로 강제
    import torch
    if isinstance(val, torch.Tensor):
        val = val.tolist()
    if isinstance(val, (list, tuple)) and len(val) == 2 and all(isinstance(x, (int, float)) for x in val):
        return [[1, val[0], val[1]]]
    if isinstance(val, list):
        out = []
        for entry in val:
            if isinstance(entry, tuple):
                entry = list(entry)
            if hasattr(entry, "tolist"):
                entry = entry.tolist()
            if isinstance(entry, list) and len(entry) in (2, 3):
                out.append([1, entry[0], entry[1]] if len(entry) == 2 else entry)
            else:
                return None
        return out
    return None

def check_with_processor(rows: List[Tuple[int, Dict[str, Any]]],
                         model_name: str,
                         limit: int = 64,
                         use_fast: bool = False) -> List[Tuple[int, str, str, str]]:
    """processor만 로드해서 grid_thw가 (t,h,w)인지 검사. 모델은 로드하지 않음."""
    if not _HF_OK:
        return [(-1, "unknown", "transformers_missing", "pip install transformers 필요")]

    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True, use_fast=use_fast)
    issues = []
    import torch

    checked = 0
    for ln, r in rows:
        if checked >= limit:
            break
        rid = row_identifier(r)
        try:
            # 이미지 준비
            if isinstance(r.get("images"), dict):
                cams = ordered_cameras(r)
                imgs = [Image.open(r["images"][c]).convert("RGB") for c in cams if c in r["images"]]
            elif "image_path" in r:
                imgs = Image.open(r["image_path"]).convert("RGB")
            else:
                issues.append((ln, rid, "no_image_key", "images|image_path 없음"))
                continue

            prompt = build_prompt_min(r)
            enc = processor(text=prompt, images=imgs, return_tensors="pt")
            # NOTE: Avoid using `or` on tensors; it triggers "Boolean value of Tensor is ambiguous".
            grid = enc.get("image_grid_thw")
            if grid is None:
                grid = enc.get("grid_thw")
            if grid is None:
                issues.append((ln, rid, "grid_missing", "processor에서 grid_thw 없음"))
                checked += 1
                continue

            n_imgs = len(imgs) if isinstance(imgs, list) else 1

            # grid는 샘플 차원이 1인 텐서거나 리스트일 수 있음
            gnorm = normalize_grid_thw(grid)
            # NOTE: Qwen2.5-VL often returns pixel_values as (total_tokens, hidden_dim) for single-sample calls.
            # For two images with grid [[1,34,46],[1,34,46]], total_tokens=2*34*46=3128 — this is expected.
            if gnorm is None:
                issues.append((ln, rid, "grid_bad_shape", f"type={type(grid).__name__}, preview={str(grid)[:120]}"))
            else:
                # 각 엔트리 3튜플인지 최종 확인
                bad = []
                for gi, entry in enumerate(gnorm):
                    if not (isinstance(entry, list) and len(entry) == 3):
                        bad.append((gi, entry))
                if bad:
                    issues.append((ln, rid, "grid_not_triplet", f"first_bad={bad[0]}"))
                if len(gnorm) != n_imgs:
                    issues.append((ln, rid, "grid_count_mismatch", f"imgs={n_imgs}, grid={len(gnorm)}"))
                for entry in gnorm:
                    try:
                        t,h,w = entry
                        if not all(isinstance(x, int) or (isinstance(x, float) and x.is_integer()) for x in entry):
                            issues.append((ln, rid, "grid_values_invalid", f"non-integer values {entry}"))
                            break
                        t=int(t); h=int(h); w=int(w)
                        if t < 1 or h <= 0 or w <= 0:
                            issues.append((ln, rid, "grid_values_invalid", f"invalid values {entry}"))
                            break
                    except Exception:
                        issues.append((ln, rid, "grid_values_invalid", f"exception on {entry}"))
                        break

            pv = enc.get("pixel_values")
            if pv is not None:
                if isinstance(pv, torch.Tensor):
                    shape = tuple(pv.shape)
                    # Accept ranks 2, 3, or 4:
                    # - rank 2: (total_image_tokens, hidden_dim)  [common in Qwen2.5-VL with multiple images]
                    # - rank 3: (C,H,W) or (total_image_tokens, hidden_dim) or (B, C, HW) depending on processor
                    # - rank 4: (B, C, H, W)
                    if len(shape) not in (2, 3, 4):
                        issues.append((ln, rid, "pixel_values_shape_unexpected", str(shape)))
                    else:
                        # If we have grid info, verify token count matches sum(t*h*w)
                        grid_raw = enc.get("image_grid_thw")
                        if grid_raw is None:
                            grid_raw = enc.get("grid_thw")
                        gnorm_for_pv = normalize_grid_thw(grid_raw) if grid_raw is not None else None
                        if gnorm_for_pv is not None:
                            grid_tokens = sum(int(t)*int(h)*int(w) for (t,h,w) in gnorm_for_pv)
                            # rank-2: (tokens, dim) should match tokens
                            if len(shape) == 2:
                                if shape[0] != grid_tokens:
                                    issues.append((ln, rid, "pixel_values_grid_mismatch", f"tokens={shape[0]} vs grid_sum={grid_tokens} shape={shape}"))
                            # rank-3 could be (tokens, dim, ?) or (C,H,W). Try to match either first or last dim with tokens; otherwise skip strict check.
                            elif len(shape) == 3:
                                if shape[0] == grid_tokens or shape[-1] == grid_tokens:
                                    pass
                            # rank-4 is raw (B,C,H,W) — token count check not applicable here.
                else:
                    issues.append((ln, rid, "pixel_values_type_unexpected", str(type(pv))))
        except Exception as e:
            issues.append((ln, rid, "processor_exception", f"{type(e).__name__}: {e}"))
        finally:
            checked += 1

    return issues

def main():
    global STRICT_IMAGES
    global CURRENT_FILE
    global CHECK_UID, CHECK_EP_ORDER
    ap = argparse.ArgumentParser("Dataset quick validator (no model load)")
    ap.add_argument("--path", required=True, help="jsonl 파일 또는 jsonl shards 디렉토리")
    ap.add_argument("--with-processor", default=None,
                    help="예: Qwen/Qwen2.5-VL-7B-Instruct (옵션, processor 만 로드)")
    ap.add_argument("--limit", type=int, default=64, help="processor 검사 샘플 상한")
    ap.add_argument("--out", type=str, default=None, help="문제 목록을 TSV로 저장할 경로")
    ap.add_argument("--strict-images", action="store_true", help="Require meta.capture.cameras to match images keys exactly when present")
    ap.add_argument("--stats", action="store_true", help="Print simple dataset stats (status distribution, camera counts) at the end")
    ap.add_argument("--only-shards", action="store_true", help="Force reading only from <path>/shards if it exists")
    ap.add_argument("--include-root", action="store_true", help="Force include top-level JSONL files even if shards/ exists (may double-count)")
    ap.add_argument("--check-uid", action="store_true", help="Check duplicate uid across scanned files")
    ap.add_argument("--check-episode-order", action="store_true", help="Check timestamp_ms non-decreasing per (chunk_id, episode_id)")
    ap.add_argument("--max-print", type=int, default=50, help="Max problems to print to console")
    args = ap.parse_args()

    STRICT_IMAGES = args.strict_images
    CHECK_UID = args.check_uid
    CHECK_EP_ORDER = args.check_episode_order

    # control file expansion with flags
    if os.path.isdir(args.path):
        shards_dir = os.path.join(args.path, "shards")
        if args.only_shards and os.path.isdir(shards_dir):
            files = glob.glob(os.path.join(shards_dir, "**", "*.jsonl"), recursive=True)
            files.sort()
        elif args.include_root:
            files = glob.glob(os.path.join(args.path, "**", "*.jsonl"), recursive=True)
            files.sort()
        else:
            files = expand_jsonl_paths(args.path)
    else:
        files = expand_jsonl_paths(args.path)

    if os.path.isdir(args.path):
        print(f"[validate] scanning dir={args.path} picked_files={len(files)} sample={os.path.basename(files[0]) if files else 'none'}")

    total = 0
    # problems: List of tuples (file, line, id, code, detail)
    problems: List[Tuple[str, int, str, str, str]] = []  # (file, line, ident, code, detail)

    # Only create histograms if stats requested
    if args.stats:
        status_hist = {}
        cam_hist = {}
    else:
        status_hist = None
        cam_hist = None

    for f in files:
        CURRENT_FILE = f
        rows = load_jsonl_lines(f)
        for ln, r in rows:
            total += 1
            if "__parse_error__" in r:
                problems.append((f, ln, "unknown", "json_parse_error", r["__parse_error__"]))
                continue
            # 구조 점검
            issues = []
            issues += check_required_keys(r)
            issues += check_images_exist_and_open(r)
            # 라벨 포맷 간단 점검
            if "target_text" in r:
                try:
                    _ = json.loads(r["target_text"])
                except Exception as e:
                    issues.append(f"target_text_not_json:{type(e).__name__}")
            elif "api_output" in r:
                ao = r["api_output"]
                for k in ("desc_1", "desc_2", "status"):
                    if k not in ao:
                        issues.append(f"api_output_missing_{k}")
            # 새로 추가한 검증들
            issues += validate_labels(r)
            issues += validate_meta_and_images(r)
            if args.check_uid or args.check_episode_order:
                issues += validate_uid_and_dups(r, ln)

            # 통계 집계
            if args.stats:
                # 상태 분포 집계
                label_status = None
                if "api_output" in r and isinstance(r.get("api_output"), dict):
                    label_status = r["api_output"].get("status")
                elif "target_text" in r:
                    try:
                        parsed = json.loads(r["target_text"])
                        if isinstance(parsed, dict):
                            label_status = parsed.get("status")
                    except Exception:
                        pass
                if label_status in {"DONE","NOT_DONE","UNCERTAIN"}:
                    status_hist[label_status] = status_hist.get(label_status, 0) + 1
                # 카메라 수 집계
                images = r.get("images")
                if isinstance(images, dict):
                    cam_hist[len(images)] = cam_hist.get(len(images), 0) + 1
                elif "image_path" in r:
                    cam_hist[1] = cam_hist.get(1, 0) + 1

            # 모은 이슈 기록
            rid = row_identifier(r)
            for code in issues:
                problems.append((f, ln, rid, code, ""))

        # 선택: processor 확인
        if args.with_processor:
            sub_issues = check_with_processor(rows, args.with_processor, limit=args.limit, use_fast=False)
            for ln, rid, code, detail in sub_issues:
                problems.append((f, ln, rid, code, detail))

    # 결과 출력
    bad = len(problems)
    print(f"[validate] files={len(files)} samples={total} problems={bad}")
    if bad:
        # 상위 max_print개만 화면에
        for i, (f, ln, rid, code, detail) in enumerate(problems[:args.max_print], 1):
            print(f"[{i:02d}] {os.path.basename(f)}:{ln}\t[{rid}]\t{code}\t{detail}")
    else:
        print("[validate] ✅ no problems found")

    # 통계 출력
    if args.stats:
        print(f"stats: uids={len(SEEN_UIDS)}, status={dict(status_hist)}, cams={dict(cam_hist)}")

    # 저장
    if args.out:
        with open(args.out, "w", encoding="utf-8") as w:
            w.write("file\tline\tid\tcode\tdetail\n")
            for f, ln, rid, code, detail in problems:
                w.write(f"{f}\t{ln}\t{rid}\t{code}\t{detail}\n")
        print(f"[validate] wrote: {args.out}")

if __name__ == "__main__":
    main()