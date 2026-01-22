# eval_video_real_time_main.py
from __future__ import annotations
import json
import logging
import time
import argparse
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path

import torch
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

"""
python eval_video_real_time_main.py \
  --taskspecs_dir ... \
  --hlp_base ... \
  --hlp_adapter ... \
  ...
"""

# Reuse existing wrappers
from evaluate.eval_HLP_LLP_v4.eval_real_time_qwen import HLPQwenV2, parse_detect_yaml
from evaluate.eval_HLP_LLP_v4.eval_real_time_pi0 import (
    LLPConfig,
    LLPRuntimeContext,
    init_llp_runtime,
    llp_step,
    llp_send_zero,
    capture_shared_observation,
    create_llp_batch_from_obs,
)
from evaluate.eval_HLP_LLP_v4.eval_real_time_main import _load_taskspecs_from_group, _to_pil_from_tensor, _device_move_batch, _make_dummy_image
from utils_keyboard import init_keyboard_listener

# Import Video-Specific Utils
from utils_video_batches import (
    make_video_detect_prompt,
    make_video_update_prompt,
    create_video_detect_batch,
    create_video_update_batch,
)

logger = logging.getLogger(__name__)

def concat_images_horizontally(pil_images):
    # PIL → numpy
    imgs = [np.array(img) for img in pil_images]

    # 높이를 맞추기 (최대 높이 기준)
    max_h = max(img.shape[0] for img in imgs)

    padded_imgs = []
    for img in imgs:
        h, w, c = img.shape
        if h < max_h:
            pad = np.zeros((max_h - h, w, c), dtype=img.dtype)
            img = np.vstack([img, pad])
        padded_imgs.append(img)

    return np.hstack(padded_imgs)

# --- Helper for parsing simplified Update output ---
def parse_video_update_yaml(text: str) -> str:
    """Video Baseline Update only returns Action_Command"""
    import yaml
    import re

    # Try safe load
    try:
        d = yaml.safe_load(text.strip().replace("```yaml", "").replace("```", ""))
        if isinstance(d, dict) and "Action_Command" in d:
            return str(d["Action_Command"]).strip()
    except:
        pass

    # Fallback regex
    m = re.search(r"Action_Command:\s*(.+)", text)
    if m:
        return m.group(1).strip()
    return ""


def run_video_detect(
        hlp: HLPQwenV2,
        obs_pil: Image.Image,
        task_text: str,
        action_command: str,
        event_list: str,
) -> Tuple[bool, str, float]:
    user_d = make_video_detect_prompt(task_text, action_command, event_list)
    batch_d = create_video_detect_batch(hlp.processor, obs_pil, user_d)
    batch_d = _device_move_batch(batch_d, hlp.model.device)

    t0 = time.time()
    out_text = hlp._generate_text(batch_d, hlp.max_new_tokens_detect)  # Reuse internal gen
    dt = time.time() - t0

    print("\n[DETECT RAW]", out_text)
    detected, event = parse_detect_yaml(out_text)
    return detected, event, dt


def run_video_update(
        hlp: HLPQwenV2,
        history_images: List[Image.Image],
        task_text: str,
        prev_action: str,
        allowed_actions: str,
        event: str,
) -> Tuple[str, float]:
    user_u = make_video_update_prompt(task_text, prev_action, event, allowed_actions)
    batch_u = create_video_update_batch(hlp.processor, history_images, user_u)
    batch_u = _device_move_batch(batch_u, hlp.model.device)

    t0 = time.time()
    out_text = hlp._generate_text(batch_u, hlp.max_new_tokens_update)
    dt = time.time() - t0

    print("\n[UPDATE RAW]", out_text)
    next_action = parse_video_update_yaml(out_text)
    return next_action, dt


def eval_video_real_time_main(
        hlp: HLPQwenV2,
        llp_cfg: LLPConfig,
        specs: Dict[str, Any],
        task_group: str = "1",
):
    llp_ctx: LLPRuntimeContext = init_llp_runtime(llp_cfg)
    listener, kstate = init_keyboard_listener(task_group)

    # --- Runtime State for Video HeLM ---
    current_task_id: Optional[str] = None
    current_inter_idx: int = 0
    global_instruction: Optional[str] = None

    # State: Visual History & Last Action
    visual_memory: List[Image.Image] = []
    current_action: str = "None"

    last_obs_pil: Optional[Image.Image] = None
    step = 0
    t_start = time.time()

    try:
        while True:
            # 1. Reset / Zero
            if kstate["set_zero"]:
                llp_send_zero(llp_ctx)
                kstate["set_zero"] = False

            if kstate["reset_episode"]:
                current_task_id = None
                visual_memory = []
                current_action = "None"
                kstate["reset_episode"] = False
                logger.info("[MAIN] Episode Reset (Memory Cleared)")
                llp_send_zero(llp_ctx)
                time.sleep(1)

            # 2. Task Selection (Init)
            sel_tid = kstate.get("selected_task_id", None)
            if sel_tid is not None:
                kstate["selected_task_id"] = None
                if sel_tid in specs:
                    spec = specs[sel_tid]
                    current_task_id = sel_tid
                    current_inter_idx = 0
                    global_instruction = spec.task_text[0]
                    current_action = "None"

                    # [Init] Start Frame Capture
                    logger.info(f"[MAIN] Task Selected: {sel_tid}. Capturing Start Frame...")

                    # Capture fresh frame for start
                    _, obs_img_t, _ = capture_shared_observation(
                        llp_ctx.piper, llp_ctx.table_rs_cam, llp_ctx.wrist_rs_cam, llp_ctx.cfg.use_devices
                    )
                    obs_pil = _to_pil_from_tensor(obs_img_t) if obs_img_t is not None else _make_dummy_image()

                    # Init Memory: [StartFrame]
                    visual_memory = [obs_pil]

                    logger.info(f"[Update] Initial update ...")

                    # Initial Update
                    current_action, dt = run_video_update(
                        hlp, visual_memory, global_instruction,
                        prev_action="None", allowed_actions=spec.allowed_actions,
                        event="task initialized"
                    )
                    logger.info(f"[MAIN] Init Action: {current_action}")

            # 3. Inter-Episode Transition
            if kstate["next_inter"]:
                kstate["next_inter"] = False
                if current_task_id:
                    spec = specs[current_task_id]
                    if current_inter_idx + 1 < len(spec.task_text):
                        current_inter_idx += 1
                        global_instruction = spec.task_text[current_inter_idx]
                        logger.info(f"[MAIN] Next Stage: {global_instruction}")

                        # Task Changed -> Trigger Update with current memory
                        # (We don't add a new image here, just re-evaluate history with new task text)
                        current_action, dt = run_video_update(
                            hlp, visual_memory, global_instruction,
                            prev_action=current_action, allowed_actions=spec.allowed_actions,
                            event="task changed"
                        )

            # Idle Check
            if current_task_id is None:
                print("nothing to do")
                time.sleep(2)
                continue

            # 4. Main Loop
            # Capture
            state, obs_img_t, _wrist_img_t = capture_shared_observation(
                llp_ctx.piper, llp_ctx.table_rs_cam, llp_ctx.wrist_rs_cam, llp_ctx.cfg.use_devices
            )
            if obs_img_t is None: continue

            obs_pil = _to_pil_from_tensor(obs_img_t)
            # Optional: Debug View
            # plt.imshow(obs_pil); plt.show()

            plt.figure()
            plt.imshow(obs_pil)
            plt.title(f"[UPDATE] image for debug | table step={step}")
            plt.axis("off")
            plt.show()

            # A. DETECT (Current Frame)
            spec = specs[current_task_id]
            detected, event_str, _ = run_video_detect(
                hlp, obs_pil, global_instruction, current_action, spec.event_list
            )

            # B. UPDATE (If detected)
            if detected:
                logger.info(f"[MAIN] Event Detected: {event_str}")
                llp_send_zero(llp_ctx)

                # Add Keyframe to Memory
                visual_memory.append(obs_pil)

                # DEBUG Update image batch
                concat_img = concat_images_horizontally(visual_memory)
                plt.figure(figsize=(20, 5))
                plt.imshow(concat_img)
                plt.title(f"[UPDATE] visual memory ({len(visual_memory)} frames)")
                plt.axis("off")
                plt.show()

                logger.info(f"[MEMORY] History Size: {len(visual_memory)}")

                # Predict Next Action
                current_action, _ = run_video_update(
                    hlp, visual_memory, global_instruction,
                    prev_action=current_action, allowed_actions=spec.allowed_actions,
                    event=event_str
                )
                logger.info(f"[MAIN] New Action: {current_action}")

                # Recapture after pause
                state, obs_img_t, _wrist_img_t = capture_shared_observation(
                    llp_ctx.piper, llp_ctx.table_rs_cam, llp_ctx.wrist_rs_cam, llp_ctx.cfg.use_devices
                )

            # C. EXECUTE (LLP)
            cmd = current_action
            if cmd and cmd.lower() not in ["done", "none", "wait"]:
                # print(f"[LLP] Executing: {cmd}")
                llp_batch = create_llp_batch_from_obs(state, obs_img_t, _wrist_img_t, cmd)
                llp_step(llp_ctx, cmd, batch=llp_batch)
            else:
                pass  # Idle

            logger.info(f"--------------------------------[Main] Loop done --------------------------------------")

            step += 1

    finally:
        try:
            listener.stop()
        except:
            pass


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    p = argparse.ArgumentParser()
    p.add_argument("--taskspecs_dir", type=str, required=True)
    p.add_argument("--task_group", type=str, required=True)
    p.add_argument("--hlp_base", type=str, required=True)
    p.add_argument("--hlp_adapter", type=str, required=True)
    p.add_argument("--llp_model_path", type=str, required=True)
    p.add_argument("--dataset_root", type=str, default=None)
    p.add_argument("--dataset_repo_id", type=str, default=None)
    p.add_argument("--use_devices", default=True, action="store_true")
    args = p.parse_args()

    # Load Specs
    specs = _load_taskspecs_from_group(args.taskspecs_dir, args.task_group)

    # Init HLP (Qwen)
    hlp = HLPQwenV2(
        base_model_path=args.hlp_base,
        adapter_path=args.hlp_adapter,
        load_in_4bit=True
    )

    # Init LLP Config
    from configs.default import DatasetConfig

    llp_cfg = LLPConfig(
        train_dataset=DatasetConfig(repo_id=args.dataset_repo_id, root=args.dataset_root),
        policy_path=args.llp_model_path,
        use_devices=args.use_devices,
    )

    eval_video_real_time_main(hlp, llp_cfg, specs, args.task_group)