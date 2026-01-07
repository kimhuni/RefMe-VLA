import os
import json

"""
python helm_datasets_v4/validate_helm_simple.py --root /data/ghkim/helm_data/press_button_N_times/jsonl_v4/merged/all_train.jsonl
"""

# ---------- 설정 ----------
REQUIRED_COMMON_KEYS = {
    "uid",
    "task_id",
    "mode"
}

REQUIRED_DETECT_KEYS = {
    "event_detected",
    "event",
    "gt_yaml"
}

REQUIRED_UPDATE_KEYS = {
    "gt_yaml"
}

REQUIRED_MEMORY_KEYS = {
    "Action_Command",
    "Working_Memory",
    "Episodic_Context"
}

# ---------- 단일 샘플 검사 ----------
def validate_sample(sample, file_path=""):
    errors = []

    # 공통
    missing = REQUIRED_COMMON_KEYS - set(sample.keys())
    if missing:
        errors.append(f"[COMMON] Missing keys {missing}")

    mode = sample.get("mode")
    if mode not in {"DETECT", "UPDATE"}:
        errors.append(f"[COMMON] Invalid mode: {mode}")

    # 이미지
    if "images" not in sample or not isinstance(sample["images"], dict):
        errors.append("[COMMON] Missing or invalid images field")

    # DETECT
    if mode == "DETECT":
        missing = REQUIRED_DETECT_KEYS - set(sample.keys())
        if missing:
            errors.append(f"[DETECT] Missing keys {missing}")

        gt = sample.get("gt_yaml", {})
        if not isinstance(gt, dict):
            errors.append("[DETECT] gt_yaml must be a dict")
        else:
            for k in ["Event_Detected", "Event"]:
                if k not in gt:
                    errors.append(f"[DETECT] gt_yaml missing {k}")

    # UPDATE
    if mode == "UPDATE":
        if "gt_yaml" not in sample:
            errors.append("[UPDATE] Missing gt_yaml")

        gt = sample.get("gt_yaml", {})
        if isinstance(gt, dict):
            missing = REQUIRED_MEMORY_KEYS - set(gt.keys())
            if missing:
                errors.append(f"[UPDATE] gt_yaml missing memory keys {missing}")

    return errors


# ---------- 폴더 단위 검사 ----------
def validate_dataset(root_dir):
    all_errors = []
    checked = 0

    for dirpath, _, filenames in os.walk(root_dir):
        for f in filenames:
            if not f.endswith(".json"):
                continue

            path = os.path.join(dirpath, f)
            try:
                with open(path, "r") as fp:
                    data = json.load(fp)
            except Exception as e:
                all_errors.append(f"[FILE] {path}: JSON load error {e}")
                continue

            checked += 1
            errs = validate_sample(data, path)
            for e in errs:
                all_errors.append(f"{path} :: {e}")

    return checked, all_errors


# ---------- 실행 ----------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser("HeLM Train/Eval Data Validator")
    parser.add_argument("--root", required=True, help="Path to train/eval data directory")
    args = parser.parse_args()

    checked, errors = validate_dataset(args.root)

    if errors:
        print("\n".join(errors))
        print(f"\n[!] Found {len(errors)} issues in {checked} samples.")
    else:
        print(f"[OK] All {checked} samples look structurally valid ✅")