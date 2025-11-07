# save as tools/view_min.py
import json, sys, glob, os
paths = sys.argv[1:] or glob.glob("/data/piper_press_the_blue_button_test/api_eval_B/shards/*.jsonl")
for p in paths:
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            try:
                o = json.loads(line)
                print(json.dumps({
                    "uid": o.get("uid"),
                    "prev_desc": o.get("prev_desc", ""),
                    "api_output": o.get("api_output", {}),
                }, ensure_ascii=False))
            except Exception:
                pass