# tools/gen_concat_recipes.py
import json
import random
import itertools
from pathlib import Path
from collections import defaultdict

"""generate recipe -> generate per task -> sample_datasets (task index) -> meta
python common/datasets/concat_dataset/generate_concat_recipe.py \
  --task_episodes_json /home/ghkim/codes/RefMe-VLA/common/datasets/concat_dataset/episode_press_the_button.json \
  --out_jsonl /data/ghkim/concat_data/press_button_in_order/press_button_GBR.jsonl \
  --task_order "press the green button" "press the blue button" "press the red button" \
  --n 100 \
  --seed 42 \
  --cap_per_episode 50
  
python common/datasets/concat_dataset/generate_concat_recipe.py \
  --task_episodes_json /home/ghkim/codes/RefMe-VLA/common/datasets/concat_dataset/episode_wipe_the_window.json \
  --out_jsonl /data/ghkim/concat_data/wipe_the_window/wipe_BMT.jsonl \
  --task_order "wipe the bottom side of the window" "wipe the middle side of the window" "wipe the top side of the window" \
  --n 100 \
  --seed 42 \
  --cap_per_episode 50

python tools/concat_dataset.py \
  --task_episodes_json task_episodes.json \
  --out_jsonl recipes_all.jsonl \
  --task_order a b c \
  --all_combinations
"""

def write_jsonl(path: str, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def load_task_episodes(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    # ensure lists
    out = {}
    for k, v in obj.items():
        if not isinstance(v, list) or not v:
            raise ValueError(f"task '{k}' must be a non-empty list")
        out[k] = [int(x) for x in v]
    return out

def gen_all_combinations(task_order, task_eps):
    pools = [task_eps[t] for t in task_order]
    for combo in itertools.product(*pools):
        yield combo

def gen_random_combinations(task_order, task_eps, n, seed=0):
    rng = random.Random(seed)
    pools = [task_eps[t] for t in task_order]
    for _ in range(n):
        yield tuple(rng.choice(pool) for pool in pools)

def main(
    task_episodes_json: str,
    out_jsonl: str,
    task_order: list,
    n: int = 1000,
    seed: int = 0,
    all_combinations: bool = False,
    start_new_episode_index: int = 0,
    boundary_noop_frames: int = 0,
    drop_last_frame_each_part: bool = True,
    cap_per_episode: int | None = None,
):
    task_eps = load_task_episodes(task_episodes_json)

    # validate task_order
    for t in task_order:
        if t not in task_eps:
            raise ValueError(f"task_order includes '{t}' but not found in {task_episodes_json}")

    # choose generator
    if all_combinations:
        combos = gen_all_combinations(task_order, task_eps)
    else:
        combos = gen_random_combinations(task_order, task_eps, n=n, seed=seed)

    usage = defaultdict(int)
    rows = []
    new_ep = start_new_episode_index

    for combo in combos:
        # optional: cap usage
        if cap_per_episode is not None:
            ok = True
            for t, ep_id in zip(task_order, combo):
                key = (t, ep_id)
                if usage[key] >= cap_per_episode:
                    ok = False
                    break
            if not ok:
                continue

        parts = [{"task": t, "episode_id": int(ep_id)} for t, ep_id in zip(task_order, combo)]
        row = {
            "new_episode_index": int(new_ep),
            "parts": parts,
            "boundary_policy": {
                "noop_frames": int(boundary_noop_frames),
                "drop_last_frame_each_part": bool(drop_last_frame_each_part),
            },
        }
        rows.append(row)

        if cap_per_episode is not None:
            for t, ep_id in zip(task_order, combo):
                usage[(t, ep_id)] += 1

        new_ep += 1

        # for random mode, stop when we have n
        if (not all_combinations) and len(rows) >= n:
            break

    if (not all_combinations) and len(rows) < n:
        print(f"[WARN] Only generated {len(rows)} recipes (target={n}). "
              f"Try increasing candidate pools or cap_per_episode.")

    write_jsonl(out_jsonl, rows)
    print(f"Wrote {len(rows)} recipes to {out_jsonl}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--task_episodes_json", required=True)
    ap.add_argument("--out_jsonl", required=True)
    ap.add_argument("--task_order", nargs="+", required=True, help="e.g. a b c")
    ap.add_argument("--n", type=int, default=1000, help="num recipes (random mode)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--all_combinations", action="store_true", default=False)
    ap.add_argument("--start_new_episode_index", type=int, default=0)
    ap.add_argument("--boundary_noop_frames", type=int, default=0)
    ap.add_argument("--drop_last_frame_each_part", action="store_true", default=True)
    ap.add_argument("--no_drop_last_frame_each_part", action="store_false", dest="drop_last_frame_each_part")
    ap.add_argument("--cap_per_episode", type=int, default=None, help="optional usage cap per (task, episode)")
    args = ap.parse_args()

    main(
        task_episodes_json=args.task_episodes_json,
        out_jsonl=args.out_jsonl,
        task_order=args.task_order,
        n=args.n,
        seed=args.seed,
        all_combinations=args.all_combinations,
        start_new_episode_index=args.start_new_episode_index,
        boundary_noop_frames=args.boundary_noop_frames,
        drop_last_frame_each_part=args.drop_last_frame_each_part,
        cap_per_episode=args.cap_per_episode,
    )
