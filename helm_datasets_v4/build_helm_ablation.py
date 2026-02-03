from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from helm_datasets_v4.core.registry import load_all_taskspecs
from helm_datasets_v4.core.spec import TaskSpecV4
from helm_datasets_v4.core.data_index import DataEpisodeV4, iter_all_episodes, load_data_episode
from helm_datasets_v4.core.io_utils import frame_path
from helm_datasets_v4.core.templates import dump_yaml, render_memory_one_line

"""
HeLM v4 (Ablation): unified STEP mode
- Each row outputs YAML with 4 keys:
  Event_Detected, Action_Command, Working_Memory, Episodic_Context
- If Event_Detected=false: memory MUST be copied EXACTLY from input Memory (prev_mem)
- If Event_Detected=true: memory MUST be updated to the next memory state (curr_mem)

Labels for balanced sampling (keep legacy names):
- detect_neg: non-event frames (identity update)
- update_intra: event frames for intra step (prev->curr)
- update_transition: init + inter-stage transition (empty->mem[0][0], (0,last)->(1,0))+

export PYTHONPATH=$(pwd)
python -m helm_datasets_v4.build_helm_unified_step \
  --data_root "/data/ghkim/helm_data/pick_place_press" \
  --out_root "/data/ghkim/helm_data/helm_v4_ablation_step" \
  --taskspecs_dir "/home/ghkim/codes/RefMe-VLA/helm_datasets_v4/taskspecs/pick_place_press" \
  --shard_size 5000
"""

# fixed early frame index position (avoid black frame 0)
TRANSITION_FRAME_POS = 10


# ------------------------------
# Filters / split / io helpers
# ------------------------------
def episode_matches_filter(ep_meta: Dict[str, Any], flt: Dict[str, Any]) -> bool:
    for k, v in flt.items():
        if ep_meta.get(k) != v:
            return False
    return True


def split_episodes(
    pairs: List[Tuple[str, str]],
    val_ratio: float,
    seed: int,
) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    # Deterministic: sort then shuffle with seed
    pairs = sorted(list(pairs))
    rnd = random.Random(seed)
    rnd.shuffle(pairs)
    n_val = int(round(len(pairs) * val_ratio))
    val = pairs[:n_val]
    train = pairs[n_val:]
    return train, val


def shard_write_jsonl(rows: List[Dict[str, Any]], out_dir: Path, prefix: str, shard_size: int) -> List[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: List[Path] = []
    for i in range(0, len(rows), shard_size):
        shard = rows[i:i + shard_size]
        p = out_dir / f"{prefix}-{i // shard_size:05d}.jsonl"
        with p.open("w", encoding="utf-8") as f:
            for r in shard:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        paths.append(p)
    return paths


def make_uid(task_id: str, chunk: str, episode: str, split: str, kind: str, inter_idx: int, step_idx: int, frame_id: int) -> str:
    # kind: init / transition / intra / neg
    return f"{task_id}@{chunk}-{episode}-{split}-{kind}-i{inter_idx}-s{step_idx}-f{frame_id:06d}"


def select_transition_frame_id(ep: DataEpisodeV4) -> Optional[int]:
    if not ep.frame_ids:
        return None
    pos = TRANSITION_FRAME_POS
    if pos >= len(ep.frame_ids):
        pos = len(ep.frame_ids) - 1
    return ep.frame_ids[pos]


# ------------------------------
# Unified STEP system prompt
# ------------------------------
def make_step_prompt(
    task_text: str,
    prev_mem: Dict[str, Any],
    *,
    llp_commands: str,
    n_images: int,
    event_list: Optional[str] = None,
) -> str:
    """
    Single prompt for unified STEP mode (detect+update).
    Output YAML must have EXACTLY these keys:
      - Event_Detected: boolean
      - Action_Command: string
      - Working_Memory: string
      - Episodic_Context: string
    """
    # Note: We intentionally do NOT ask to output the event name.
    event_list_block = f"\nEvent_List:\n{event_list}\n" if event_list else ""
    mem_line = render_memory_one_line(prev_mem)

    # We keep images placeholders consistent with templates
    img_placeholder = "<image_table>" if n_images == 1 else "<image_table>\n<image_wrist>"

    return (
        "Role: Robot arm HeLM Step Updater (unified DETECT+UPDATE mode).\n"
        "Goal: From the current image and memory, decide whether the current stage is completed, "
        "and output the correct memory state after this step.\n\n"

        "Inputs:\n"
        "- Task: current stage instruction.\n"
        "- Allowed_Action_Commands: valid low-level commands.\n"
        "- Memory: {Action_Command, Working_Memory, Episodic_Context} (current state; may be None at the start).\n"
        "- Images.\n\n"

        "Event detection rules:\n"
        "- Event_Detected=true ONLY if the stage completion (or clearly post-completion state) is visibly confirmed in the image.\n"
        "- If there is any uncertainty (partial progress, occlusion, ambiguous state) -> Event_Detected=false.\n"
        "- Use Task as the primary criterion; use Memory only to interpret what counts as completion.\n\n"

        "Memory update rules (CRITICAL):\n"
        "1) If Memory is None (initial state):\n"
        "   - You MUST initialize the memory by outputting the FIRST memory state for this stage.\n"
        "   - Set Event_Detected=true.\n"
        "2) If Memory is NOT None and Event_Detected=false:\n"
        "   - You MUST copy the input Memory EXACTLY (verbatim) to the output.\n"
        "   - Do NOT change Action_Command, Working_Memory, or Episodic_Context in any way.\n"
        "3) If Event_Detected=true:\n"
        "   - You MUST output the NEXT memory state for this stage.\n"
        "   - Progress must advance monotonically (no partial or intermediate states).\n"
        "   - Episodic_Context may change ONLY if the next state changes it.\n\n"

        "Field semantics:\n"
        "- Action_Command: the next low-level command (must be from Allowed_Action_Commands).\n"
        "- Working_Memory: encodes intra-stage progress (monotonic, task-consistent).\n"
        "- Episodic_Context: accumulated cross-stage history (do not rewrite arbitrarily).\n\n"

        "Output format:\n"
        "- Output YAML ONLY.\n"
        "- Output EXACTLY these keys:\n"
        "  Event_Detected, Action_Command, Working_Memory, Episodic_Context\n"
        "- No extra text or explanation.\n\n"

        f"Task: {task_text}\n"
        f"Allowed_Action_Commands:\n{llp_commands}\n"
        f"Memory: {mem_line}\n"
        f"Images: {img_placeholder}\n"
    )


# ------------------------------
# Row builders (unified STEP)
# ------------------------------
def build_step_rows_for_episode_step(
    spec: TaskSpecV4,
    ep: DataEpisodeV4,
    data_root: Path,
    fps_out: int,
    n_images: int,
    split: str,
    inter_idx: int,
    step_idx: int,
    make_transition_in_step0: bool = True,
) -> List[Dict[str, Any]]:
    """
    Unified STEP rows for a given (inter_idx, step_idx).
    Labels:
      - update_transition: init + inter transition (Event_Detected=true)
      - update_intra: event frames for intra step (Event_Detected=true)
      - detect_neg: non-event frames (Event_Detected=false, identity memory)
    """
    rows: List[Dict[str, Any]] = []

    task_text = spec.task_text[inter_idx]
    event_list = getattr(spec, "event_list", None)

    # Common prev memory (current stage state)
    prev_mem_stage = spec.memory_grid[inter_idx][step_idx]

    # -------- (0) init: EMPTY -> memory_grid[0][0] --------
    if inter_idx == 0 and step_idx == 0:
        prev_mem = {}
        curr_mem = spec.memory_grid[0][0]
        frame_id = select_transition_frame_id(ep)
        if frame_id is not None:
            prompt = make_step_prompt(
                task_text,
                prev_mem,
                llp_commands=spec.llp_commands,
                n_images=n_images,
                event_list=event_list,
            )
            gt_yaml = {
                "Event_Detected": True,
                "Action_Command": curr_mem["Action_Command"],
                "Working_Memory": curr_mem["Working_Memory"],
                "Episodic_Context": curr_mem["Episodic_Context"],
            }
            gt_text = dump_yaml(gt_yaml)

            images = {"table": str(frame_path(data_root, fps_out, ep.chunk, ep.episode, "table", frame_id))}
            if n_images == 2:
                images["wrist"] = str(frame_path(data_root, fps_out, ep.chunk, ep.episode, "wrist", frame_id))

            rows.append({
                "uid": make_uid(spec.task_id, ep.chunk, ep.episode, split, "init", inter_idx, step_idx, frame_id),
                "task_id": spec.task_id,
                "mode": "STEP",
                "label": "update_transition",
                "kind": "init",
                "event_detected": True,
                "chunk": ep.chunk,
                "episode": ep.episode,
                "inter": inter_idx,
                "step": step_idx,
                "frame_id": frame_id,
                "images": images,
                "user_prompt": prompt,
                "gt_text": gt_text,
                "gt_yaml": gt_yaml,
                "meta": {
                    "data_episode_tasks": ep.meta.get("tasks"),
                    "episode_index": ep.meta.get("episode_index"),
                },
            })

    # -------- (1) inter transition: (0,last)->(1,0) --------
    if make_transition_in_step0 and inter_idx == 1 and step_idx == 0:
        prev_mem, curr_mem = spec.transition_prev_curr()
        frame_id = select_transition_frame_id(ep)
        if frame_id is not None:
            prompt = make_step_prompt(
                task_text,
                prev_mem,
                llp_commands=spec.llp_commands,
                n_images=n_images,
                event_list=event_list,
            )
            gt_yaml = {
                "Event_Detected": True,
                "Action_Command": curr_mem["Action_Command"],
                "Working_Memory": curr_mem["Working_Memory"],
                "Episodic_Context": curr_mem["Episodic_Context"],
            }
            gt_text = dump_yaml(gt_yaml)

            images = {"table": str(frame_path(data_root, fps_out, ep.chunk, ep.episode, "table", frame_id))}
            if n_images == 2:
                images["wrist"] = str(frame_path(data_root, fps_out, ep.chunk, ep.episode, "wrist", frame_id))

            # rows.append({
            #     "uid": make_uid(spec.task_id, ep.chunk, ep.episode, split, "transition", inter_idx, step_idx, frame_id),
            #     "task_id": spec.task_id,
            #     "mode": "STEP",
            #     "label": "update_transition",
            #     "kind": "transition",
            #     "event_detected": True,
            #     "chunk": ep.chunk,
            #     "episode": ep.episode,
            #     "inter": inter_idx,
            #     "step": step_idx,
            #     "frame_id": frame_id,
            #     "images": images,
            #     "user_prompt": prompt,
            #     "gt_text": gt_text,
            #     "gt_yaml": gt_yaml,
            #     "meta": {
            #         "data_episode_tasks": ep.meta.get("tasks"),
            #         "episode_index": ep.meta.get("episode_index"),
            #     },
            # })

    # -------- (2) intra positive updates over ALL event frames --------
    prev_mem, curr_mem = spec.prev_curr_for_step(inter_idx, step_idx)

    prompt_pos = make_step_prompt(
        task_text,
        prev_mem,
        llp_commands=spec.llp_commands,
        n_images=n_images,
        event_list=event_list,
    )
    gt_yaml_pos = {
        "Event_Detected": True,
        "Action_Command": curr_mem["Action_Command"],
        "Working_Memory": curr_mem["Working_Memory"],
        "Episodic_Context": curr_mem["Episodic_Context"],
    }
    gt_text_pos = dump_yaml(gt_yaml_pos)

    for frame_id in ep.event_frame_ids:
        images = {"table": str(frame_path(data_root, fps_out, ep.chunk, ep.episode, "table", frame_id))}
        if n_images == 2:
            images["wrist"] = str(frame_path(data_root, fps_out, ep.chunk, ep.episode, "wrist", frame_id))

        rows.append({
            "uid": make_uid(spec.task_id, ep.chunk, ep.episode, split, "intra", inter_idx, step_idx, frame_id),
            "task_id": spec.task_id,
            "mode": "STEP",
            "label": "update_intra",
            "kind": "intra",
            "event_detected": True,
            "chunk": ep.chunk,
            "episode": ep.episode,
            "inter": inter_idx,
            "step": step_idx,
            "frame_id": frame_id,
            "event_frame_ids": ep.event_frame_ids,
            "images": images,
            "user_prompt": prompt_pos,
            "gt_text": gt_text_pos,
            "gt_yaml": gt_yaml_pos,
            "meta": {
                "data_episode_tasks": ep.meta.get("tasks"),
                "episode_index": ep.meta.get("episode_index"),
            },
        })

    # -------- (3) detect_neg rows: reuse legacy detect builder policy (ALL non-event frames) --------
    # Event frames are excluded to avoid duplicates with update_intra rows.
    event_set = set(ep.event_frame_ids)
    prompt_neg = make_step_prompt(
        task_text,
        prev_mem_stage,  # current step memory
        llp_commands=spec.llp_commands,
        n_images=n_images,
        event_list=event_list,
    )
    gt_yaml_neg = {
        "Event_Detected": False,
        "Action_Command": prev_mem_stage["Action_Command"],
        "Working_Memory": prev_mem_stage["Working_Memory"],
        "Episodic_Context": prev_mem_stage["Episodic_Context"],
    }
    gt_text_neg = dump_yaml(gt_yaml_neg)

    for frame_id in ep.frame_ids:
        if frame_id in event_set:
            continue
        images = {"table": str(frame_path(data_root, fps_out, ep.chunk, ep.episode, "table", frame_id))}
        if n_images == 2:
            images["wrist"] = str(frame_path(data_root, fps_out, ep.chunk, ep.episode, "wrist", frame_id))

        rows.append({
            "uid": make_uid(spec.task_id, ep.chunk, ep.episode, split, "neg", inter_idx, step_idx, frame_id),
            "task_id": spec.task_id,
            "mode": "STEP",
            "label": "detect_neg",
            "kind": "neg",
            "event_detected": False,
            "chunk": ep.chunk,
            "episode": ep.episode,
            "inter": inter_idx,
            "step": step_idx,
            "frame_id": frame_id,
            "event_frame_ids": ep.event_frame_ids,
            "images": images,
            "user_prompt": prompt_neg,
            "gt_text": gt_text_neg,
            "gt_yaml": gt_yaml_neg,
            "meta": {
                "data_episode_tasks": ep.meta.get("tasks"),
                "episode_index": ep.meta.get("episode_index"),
            },
        })

    return rows


def build_for_task(
    spec: TaskSpecV4,
    out_root: Path,
    data_root: Path,
    fps_out: int,
    n_images: int,
    val_ratio: float,
    seed: int,
    shard_size: int,
) -> None:
    all_pairs = list(iter_all_episodes(data_root, fps_out=fps_out))
    train_pairs, val_pairs = split_episodes(all_pairs, val_ratio=val_ratio, seed=seed)

    def load_pairs(pairs: List[Tuple[str, str]]) -> List[DataEpisodeV4]:
        eps: List[DataEpisodeV4] = []
        for chunk, episode in pairs:
            ep = load_data_episode(data_root, fps_out, chunk, episode, use_wrist=(n_images == 2))
            eps.append(ep)
        return eps

    train_eps = load_pairs(train_pairs)
    val_eps = load_pairs(val_pairs)

    task_out = out_root / "jsonl_v4" / spec.task_id
    step_out = task_out / "step"

    def build_for_split(eps: List[DataEpisodeV4], split_name: str) -> List[Dict[str, Any]]:
        out_rows: List[Dict[str, Any]] = []
        for inter_idx in range(spec.inter + 1):
            for step_idx in range(spec.intra[inter_idx]):
                flt = spec.episode_filters[inter_idx][step_idx]
                for ep in eps:
                    if not episode_matches_filter(ep.meta, flt):
                        continue
                    out_rows.extend(build_step_rows_for_episode_step(
                        spec=spec,
                        ep=ep,
                        data_root=data_root,
                        fps_out=fps_out,
                        n_images=n_images,
                        split=split_name,
                        inter_idx=inter_idx,
                        step_idx=step_idx,
                        make_transition_in_step0=True,
                    ))
        return out_rows

    train_rows = build_for_split(train_eps, "train")
    val_rows = build_for_split(val_eps, "val")

    shard_write_jsonl(train_rows, step_out, "train", shard_size)
    shard_write_jsonl(val_rows, step_out, "val", shard_size)

    meta = {
        "task_id": spec.task_id,
        "fps_out": fps_out,
        "n_images": n_images,
        "val_ratio": val_ratio,
        "seed": seed,
        "transition_frame_pos": TRANSITION_FRAME_POS,
        "note": "v4 ablation: unified STEP rows with labels detect_neg/update_intra/update_transition. Event name not output; event_detected controls whether memory changes.",
    }
    (task_out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--out_root", type=str, required=True)
    p.add_argument("--taskspecs_dir", type=str, required=True)
    p.add_argument("--tasks", type=str, nargs="*", default=None, help="optional task_id list. if omitted, build ALL tasks in taskspecs_dir")
    p.add_argument("--fps_out", type=int, default=5)
    p.add_argument("--n_images", type=int, default=1, choices=[1, 2])
    p.add_argument("--val_ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--shard_size", type=int, default=5000)
    args = p.parse_args()

    data_root = Path(args.data_root)
    out_root = Path(args.out_root)
    taskspecs_dir = Path(args.taskspecs_dir)

    specs = load_all_taskspecs(taskspecs_dir)
    task_ids = args.tasks if args.tasks else sorted(specs.keys())

    for task_id in task_ids:
        if task_id not in specs:
            raise KeyError(f"task_id not found in taskspecs_dir: {task_id}")

        build_for_task(
            specs[task_id],
            out_root=out_root,
            data_root=data_root,
            fps_out=args.fps_out,
            n_images=args.n_images,
            val_ratio=args.val_ratio,
            seed=args.seed,
            shard_size=args.shard_size,
        )

    print("[v4 ablation] build complete.")


if __name__ == "__main__":
    main()
