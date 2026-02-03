# eval_real_time_main.py v4
from __future__ import annotations
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path

from eval_real_time_smolvlm import HLPSmolVLM
from eval_real_time_smolvla import (
    LLPConfig,
    LLPRuntimeContext,
    init_llp_runtime,
    llp_step,
    llp_send_zero,
    capture_shared_observation,
    create_llp_batch_from_obs,
)

from utils_keyboard import init_keyboard_listener
from utils_batches import (
    create_hlp_detect_batch,
    create_hlp_update_batch,
)

from helm_datasets_v4.core.templates import make_detect_prompt, make_update_prompt

logger = logging.getLogger(__name__)

"""
python eval_real_time_main.py \
  --taskspecs_dir /Users/ghkim/codes/RefMe-VLA/helm_datasets_v3/taskspecs \
  --task_group press_button_N_times_M_times_total \
  --hlp_base /path/to/Qwen2.5-VL \
  --hlp_adapter /path/to/adapter \
  --llp_model_path /ckpt/pi0 \
  --dataset_root /path/to/lerobot \
  --use_devices
"""

@dataclass
class TaskSpecRuntime:
    task_id: str
    task_text: List[str]                  # len 1 or 2
    init_memory: Dict[str, Any]           # dict includes Action_Command
    allowed_actions: str            # list of allowed action commands
    event_list: str


def _make_dummy_image(size: Tuple[int, int] = (224, 224)) -> List[Image.Image]:
    """
    캡처 이미지가 아직 없을 때만 fallback으로 쓰는 black 이미지들.
    (가능하면 실시간에서는 캡처 이미지 사용을 권장)
    """
    return Image.new("RGB", size, color=(0, 0, 0))

def _load_taskspecs_from_group(taskspecs_dir: str, task_group: str) -> Dict[str, TaskSpecRuntime]:
    """
    taskspecs_dir/<task_group> 아래 모든 json 재귀 로드
    """
    root = Path(taskspecs_dir) / task_group
    if not root.exists():
        raise FileNotFoundError(f"Task group dir not found: {root}")

    out: Dict[str, TaskSpecRuntime] = {}
    for p in sorted(root.rglob("*.json")):
        try:
            raw = json.loads(p.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning(f"[TASKSPEC] failed to read {p}: {e}")
            continue

        tid = str(raw.get("task_id", "")).strip()
        if not tid:
            logger.warning(f"[TASKSPEC] missing task_id in {p}")
            continue

        tt = raw.get("task_text", [])
        if isinstance(tt, str):
            tt = [tt]
        if not isinstance(tt, list) or not tt:
            logger.warning(f"[TASKSPEC] invalid task_text in {p} (task_id={tid})")
            continue
        task_text = [str(x).strip() for x in tt if str(x).strip()]

        mem_grid = raw.get("memory_grid", None)
        init_mem = mem_grid[0][0]
        if not isinstance(init_mem, dict):
            logger.warning(f"[TASKSPEC] init_memory must be dict in {p} (task_id={tid})")
            mem_grid = {}

        # allowed actions: llp_command_list 우선, 없으면 allowed_actions
        allowed_actions = raw.get("llp_commands", None)
        event_list = raw.get("event_list", "none\ndone")

        # 필수: init_memory에 Action_Command 포함 (너가 확정한 (i))
        if "Action_Command" not in init_mem:
            logger.warning(f"[TASKSPEC] init_memory missing Action_Command in {p} (task_id={tid})")

        out[tid] = TaskSpecRuntime(
            task_id=tid,
            task_text=task_text,
            init_memory=init_mem,
            allowed_actions=allowed_actions,
            event_list=event_list,
        )

    logger.info(f"[TASKSPEC] loaded {len(out)} specs from group='{task_group}' at {root}")
    return out


def _to_pil_from_tensor(img_t) -> Image.Image:
    """
    Supports:
      - torch.Tensor / np.ndarray
      - shapes: (H,W,3), (3,H,W), (1,H,W), (B,C,H,W), (T,C,H,W), etc.
    Returns RGB PIL.
    """
    if isinstance(img_t, torch.Tensor):
        arr = img_t.detach().cpu().numpy()
    else:
        arr = np.array(img_t)

    # 1) squeeze trivial dims (common: (1,1,H,W))
    # but be careful not to squeeze away H/W
    while arr.ndim >= 4 and arr.shape[0] == 1:
        arr = arr[0]
    while arr.ndim >= 4 and arr.shape[0] != 1 and arr.shape[0] not in (3,):
        # If still 4D like (T,C,H,W), take first frame
        arr = arr[0]

    # Now handle 3D / 2D
    if arr.ndim == 3:
        # CHW -> HWC
        if arr.shape[0] in (1, 3) and arr.shape[-1] not in (1, 3):
            arr = np.transpose(arr, (1, 2, 0))

        # if single channel -> expand to 3
        if arr.shape[-1] == 1:
            arr = np.repeat(arr, 3, axis=-1)

        # if still not 3 channels, try best-effort
        if arr.shape[-1] != 3:
            raise ValueError(f"Unsupported image shape after processing: {arr.shape}")

    elif arr.ndim == 2:
        # grayscale -> RGB
        arr = np.stack([arr, arr, arr], axis=-1)

    else:
        raise ValueError(f"Unsupported image ndim={arr.ndim}, shape={arr.shape}")

    # dtype normalize
    if arr.dtype != np.uint8:
        # sometimes float 0..1 or 0..255
        if arr.max() <= 1.0:
            arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
        else:
            arr = arr.clip(0, 255).astype(np.uint8)

    return Image.fromarray(arr, mode="RGB")


def _device_move_batch(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    out = {}
    for k, v in batch.items():
        if hasattr(v, "to"):
            out[k] = v.to(device)
        else:
            out[k] = v
    return out


def run_hlp_detect(
    hlp: HLPSmolVLM,
    obs_pil: Image.Image,
    task_text: str,
    memory: Dict[str, Any],
    event_list: Optional[str],
    n_images: int = 1,
) -> Tuple[bool, str, float]:
    """
    DETECT = prompt 생성(make_detect_prompt) + batch 생성 + forward + parse
    return: (event_detected, event_str, latency_sec)
    """
    user_d = make_detect_prompt(
        task_text=task_text,
        memory=memory,
        n_images=n_images,
        event_list=event_list,
    )
    # batch_d = create_hlp_detect_batch(hlp.processor, obs_pil, user_d)
    # batch_d = _device_move_batch(batch_d, hlp.model.device)

    t0 = time.time()
    out = hlp.detect([obs_pil], user_d)
    dt = time.time() - t0

    print("[DETECT] detect time: ", dt)

    # backward-compat: hlp.detect가 bool만 리턴하는 버전이면 event는 none
    if isinstance(out, tuple) and len(out) == 2:
        detected, event = out
    else:
        detected, event = bool(out), "none"

    if not detected:
        event = "none"
    return detected, str(event).strip(), dt


def run_hlp_update(
    hlp: HLPSmolVLM,
    obs_pil: Image.Image,
    task_text: str,
    prev_memory: Dict[str, Any],
    allowed_actions: str,
    event: str = "none",
    n_images: int = 1,
) -> Tuple[Dict[str, Any], float]:
    """
    UPDATE = prompt 생성(make_update_prompt) + batch 생성 + forward + parse
    return: (next_memory, latency_sec)
    """
    user_u = make_update_prompt(
        task_text=task_text,
        prev_memory=prev_memory,
        n_images=n_images,
        llp_commands=allowed_actions,
        event=event,
    )
    # batch_u = create_hlp_update_batch(hlp.processor, obs_pil, user_u)
    # batch_u = _device_move_batch(batch_u, hlp.model.device)

    t0 = time.time()
    upd = hlp.update([obs_pil], user_u)
    dt = time.time() - t0

    print("[UPDATE] update time: ", dt)

    next_memory = {
        "Working_Memory": upd.get("Working_Memory", ""),
        "Episodic_Context": upd.get("Episodic_Context", ""),
        "Action_Command": upd.get("Action_Command", ""),
    }
    return next_memory, dt

def eval_real_time_main_v4(
    hlp: HLPSmolVLM,
    llp_cfg: LLPConfig,
    specs: Dict[str, TaskSpecRuntime],
    task_group: str = "1",
):
    llp_ctx: LLPRuntimeContext = init_llp_runtime(llp_cfg)
    listener, kstate = init_keyboard_listener(task_group)

    # runtime states
    current_task_id: Optional[str] = None
    current_inter_idx: int = 0               # 0 or 1
    global_instruction: Optional[str] = None
    current_memory: Optional[Dict[str, Any]] = None

    last_obs_pil: Optional[Image.Image] = None

    step = 0
    t_start = time.time()

    try:
        while True:
            # robot zero
            if kstate["set_zero"]:
                llp_send_zero(llp_ctx)
                kstate["set_zero"] = False

            # episode reset (0)
            if kstate["reset_episode"]:
                current_task_id = None
                current_inter_idx = 0
                global_instruction = None
                current_memory = None
                kstate["reset_episode"] = False
                logger.info("[MAIN] episode reset -> GI=None, memory=None, inter=0")
                llp_send_zero(llp_ctx)
                time.sleep(3)

            # [Memory Init]
            sel_tid = kstate.get("selected_task_id", None)
            if sel_tid is not None:
                kstate["selected_task_id"] = None

                if sel_tid not in specs:
                    logger.warning(f"[MAIN] selected task_id not found in loaded specs: {sel_tid}")
                else:
                    new_spec = specs[sel_tid]
                    # Always use UPDATE to initialize memory for a (new) task.
                    # This applies to both None->new and prev->new.
                    prev_mem = current_memory if isinstance(current_memory, dict) else {}
                    current_task_id = sel_tid
                    current_inter_idx = 0
                    global_instruction = new_spec.task_text[0]
                    print("Global_instruction: ", global_instruction)

                    logger.info(
                        f"[MAIN] '{global_instruction}' -> UPDATE"
                    )

                    obs_pil_for_update = last_obs_pil if last_obs_pil is not None else _make_dummy_image()
                    current_memory, dt = run_hlp_update(
                        hlp=hlp,
                        obs_pil=obs_pil_for_update,
                        task_text=global_instruction,
                        prev_memory=prev_mem,
                        allowed_actions=new_spec.allowed_actions,
                        event="task initialized",
                        n_images=1,
                    )
                    logger.info(
                        f"[MAIN] task_id={sel_tid} inter=0 UPDATE@select {dt:.3f}s "
                        f"Global_Instruction='{global_instruction}' Action='{current_memory.get('Action_Command','')}'"
                    )

            # [Inter Episode Task] next inter (n): same task_id, switch to task_text[1] if exists, UPDATE 한번
            if kstate["next_inter"]:
                kstate["next_inter"] = False
                if current_task_id is None or global_instruction is None or current_memory is None:
                    logger.info("[MAIN] next_inter ignored (no active task)")
                else:
                    spec = specs.get(current_task_id, None)
                    if spec is None:
                        logger.warning("[MAIN] next_inter: current_task_id spec not found")
                    elif current_inter_idx + 1 >= len(spec.task_text):
                        logger.info("[MAIN] next_inter: no further task_text (len<=1)")
                    else:
                        prev_mem = current_memory
                        current_inter_idx += 1
                        global_instruction = spec.task_text[current_inter_idx]
                        print("New global instruction:", global_instruction)
                        obs_pil_for_update = last_obs_pil if last_obs_pil is not None else _make_dummy_image()
                        current_memory, dt = run_hlp_update(
                            hlp=hlp,
                            obs_pil=obs_pil_for_update,
                            task_text=global_instruction,
                            prev_memory=prev_mem,
                            allowed_actions=spec.allowed_actions,
                            event="task changed",
                            n_images=1,
                        )
                        logger.info(
                            f"[MAIN] next_inter -> inter={current_inter_idx} GI='{global_instruction}' "
                            f"UPDATE {dt:.3f}s Action='{current_memory.get('Action_Command','')}'"
                        )

            # idle if no task
            if current_task_id is None or global_instruction is None or current_memory is None:
                print("nothing to do")
                time.sleep(3)
                continue


            # Get Image
            state, obs_img_t, _wrist_img_t = capture_shared_observation(
                piper=llp_ctx.piper,
                table_rs_cam=llp_ctx.table_rs_cam,
                wrist_rs_cam=llp_ctx.wrist_rs_cam,
                use_devices=llp_ctx.cfg.use_devices,
                use_end_pose=True,
            )

            if obs_img_t is None:
                time.sleep(0.3)
                continue

            # 1장 고정: table 관측만 사용
            obs_pil = _to_pil_from_tensor(obs_img_t)

            plt.figure()
            plt.imshow(obs_pil)
            plt.title(f"[UPDATE] image for debug | table step={step}")
            plt.axis("off")
            plt.show()

            # UPDATE에서 재사용할 수 있게 캐시
            last_obs_pil = obs_pil

            # -------- DETECT --------
            spec = specs[current_task_id]
            detected, event_str, t_detect = run_hlp_detect(
                hlp=hlp,
                obs_pil=obs_pil,
                task_text=global_instruction,
                memory=current_memory,
                event_list=spec.event_list,
                n_images=1,
            )

            # -------- UPDATE (only if event detected) --------
            t_update = 0.0
            if detected:
                llp_send_zero(llp_ctx)
                logger.info(f"[MAIN] EVENT DETECTED -> UPDATE (event='{event_str}')")
                current_memory, t_update = run_hlp_update(
                    hlp=hlp,
                    obs_pil=obs_pil,
                    task_text=global_instruction,
                    prev_memory=current_memory,
                    allowed_actions=spec.allowed_actions,
                    event=event_str,
                    n_images=1,
                )

                state, obs_img_t, _wrist_img_t = capture_shared_observation(
                    piper=llp_ctx.piper,
                    table_rs_cam=llp_ctx.table_rs_cam,
                    wrist_rs_cam=llp_ctx.wrist_rs_cam,
                    use_devices=llp_ctx.cfg.use_devices,
                    use_end_pose=True,
                )

            event = detected  # keep existing downstream logging expectations

            # [LLP] step - Action_Command
            cmd = str(current_memory.get("Action_Command", "")).strip()

            if cmd and ((cmd != "done") and (cmd != "wait")):
                print("[LLP] executing command: ", cmd)
                llp_batch = create_llp_batch_from_obs(
                    state=state,
                    table_img=obs_img_t,
                    wrist_img=_wrist_img_t,
                    task=cmd,
                )
                t_pred, t_llp = llp_step(llp_ctx, task_text=cmd, batch=llp_batch)
            else:
                print("no action command")
                t_llp = 0.0

            step += 1
            fps = step / max(1e-6, (time.time() - t_start))
            print(f"[MAIN] current_memory = f{current_memory} \n")
            print("=========================================================[Action Done]=================================================================")

    finally:
        try:
            listener.stop()
        except Exception:
            pass


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)

    p = argparse.ArgumentParser()
    p.add_argument("--taskspecs_dir", type=str, required=True)
    p.add_argument("--task_group", type=str, required=True)

    p.add_argument("--hlp_base", type=str, required=True)
    p.add_argument("--hlp_adapter", type=str, required=True)
    p.add_argument("--hlp_device", type=str, default="cuda:0")
    p.add_argument("--hlp_attn", type=str, default="sdpa")

    # LLP args는 네 프로젝트 config에 맞게 유지
    p.add_argument("--llp_model_path", type=str, required=True)
    p.add_argument("--dataset_repo_id", type=str, default=None)
    p.add_argument("--dataset_root", type=str, default=None)
    p.add_argument("--use_devices", action="store_true")
    p.add_argument("--no_use_devices", dest="use_devices", action="store_false")
    p.set_defaults(use_devices=True)
    p.add_argument("--llp_device", type=str, default="cuda:0")
    p.add_argument("--max_steps", type=int, default=1000000)
    args = p.parse_args()

    specs = _load_taskspecs_from_group(args.taskspecs_dir, args.task_group)

    # HLP
    hlp = HLPSmolVLM(
        base_model_path=args.hlp_base,
        device=args.hlp_device,
        attn_impl=args.hlp_attn,
    )

    # LLP cfg (네 프로젝트의 DatasetConfig 경로에 맞춰 수정 필요)
    from configs.default import DatasetConfig
    llp_cfg = LLPConfig(
        train_dataset=DatasetConfig(repo_id=args.dataset_repo_id, root=args.dataset_root),
        policy_path=args.llp_model_path,
        use_devices=bool(args.use_devices),
        task="",
        max_steps=args.max_steps,
        device=args.llp_device,
    )
    eval_real_time_main_v4(
        hlp=hlp,
        llp_cfg=llp_cfg,
        specs=specs,
        task_group=args.task_group,
    )