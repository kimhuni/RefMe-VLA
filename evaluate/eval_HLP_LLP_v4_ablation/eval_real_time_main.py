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

from eval_real_time_qwen import HLPQwenV2
from eval_real_time_pi0 import (
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
    create_hlp_step_batch,
)

from utils_batches import create_hlp_step_batch, render_memory_one_line

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


def make_step_prompt(task_text: str, allowed_actions: str, memory: Optional[Dict[str, Any]]) -> str:
    mem_line = "None"
    if isinstance(memory, dict) and memory:
        mem_line = render_memory_one_line(memory)

    img_placeholder = "<image_table>"

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
        f"Allowed_Action_Commands:\n{allowed_actions}\n"
        f"Memory: {mem_line}\n"
        f"Images: {img_placeholder}\n"
    )


def run_hlp_step(
    hlp: HLPQwenV2,
    obs_pil: Image.Image,
    task_text: str,
    memory: Optional[Dict[str, Any]],
    allowed_actions: str,
) -> Tuple[bool, Dict[str, Any], float]:
    user = make_step_prompt(
        task_text=task_text,
        allowed_actions=allowed_actions,
        memory=memory,
    )
    batch = create_hlp_step_batch(hlp.processor, obs_pil, user)
    batch = _device_move_batch(batch, hlp.model.device)

    t0 = time.time()
    out = hlp.step(batch)
    dt = time.time() - t0

    detected = bool(out.get("Event_Detected", False))
    next_memory = {
        "Action_Command": out.get("Action_Command", ""),
        "Working_Memory": out.get("Working_Memory", ""),
        "Episodic_Context": out.get("Episodic_Context", ""),
    }
    return detected, next_memory, dt

def eval_real_time_main_v4(
    hlp: HLPQwenV2,
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
                    current_task_id = sel_tid
                    current_inter_idx = 0
                    global_instruction = new_spec.task_text[0]
                    print("Global_instruction: ", global_instruction)

                    # unified STEP에서는 init을 위해 memory=None으로 시작
                    current_memory = None

                    obs_pil_for_step = last_obs_pil if last_obs_pil is not None else _make_dummy_image()
                    det, next_mem, dt = run_hlp_step(
                        hlp=hlp,
                        obs_pil=obs_pil_for_step,
                        task_text=global_instruction,
                        memory=current_memory,  # None -> init
                        allowed_actions=new_spec.allowed_actions,
                    )
                    current_memory = next_mem

                    logger.info(
                        f"[MAIN] task_id={sel_tid} inter=0 STEP@select {dt:.3f}s "
                        f"event={det} GI='{global_instruction}' Action='{current_memory.get('Action_Command', '')}'"
                    )
                    current_task_id = sel_tid


            # idle if no task
            if current_task_id is None or global_instruction is None or current_memory is None:
                print(current_task_id, "and", global_instruction,"and", current_memory)
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


            global_instruction = new_spec.task_text[0]
            print("Global_instruction: ", global_instruction)

            # unified STEP에서는 init을 위해 memory=None으로 시작
            # current_memory = None

            obs_pil_for_step = last_obs_pil if last_obs_pil is not None else _make_dummy_image()
            det, next_mem, dt = run_hlp_step(
                hlp=hlp,
                obs_pil=obs_pil_for_step,
                task_text=global_instruction,
                memory=current_memory,  # None -> init
                allowed_actions=new_spec.allowed_actions,
            )
            current_memory = next_mem

            logger.info(
                f"[MAIN] task_id={sel_tid} inter=0 STEP@select {dt:.3f}s "
                f"event={det} GI='{global_instruction}' Action='{current_memory.get('Action_Command', '')}'"
            )


            print("event: ", det)
            if det:
                llp_send_zero(llp_ctx)

            else:
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
    hlp = HLPQwenV2(
        base_model_path=args.hlp_base,
        adapter_path=args.hlp_adapter,
        device=args.hlp_device,
        attn_impl=args.hlp_attn,
        load_in_4bit=True,
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