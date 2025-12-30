# eval_real_time_main.py
from __future__ import annotations
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from termcolor import colored
from PIL import Image

from eval_real_time_qwen import HLPQwenV2, DETECT_HEADER, UPDATE_HEADER
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
    build_detect_user_text,
    build_update_user_text,
    create_hlp_detect_batch,
    create_hlp_update_batch,
)

"""
python eval_real_time_main.py \
  --taskspecs_dir /path/to/helm_datasets_v2/taskspecs \
  --keymap_json /path/to/keymap_press.json \
  --hlp_base /path/to/Qwen2.5-VL \
  --hlp_adapter /path/to/qlora_adapter \
  --llp_model_path /ckpt/pi0 \
  --dataset_repo_id <...> \
  --dataset_root <...> \
  --use_devices
"""

@dataclass
class TaskRuntimeSpec:
    task_id: str
    global_instruction: str
    init_memory: Dict[str, Any]              # must include Action_Command
    allowed_actions: List[str]               # Allowed_Action_Commands


def _load_taskspecs_recursive(taskspecs_dir: str) -> Dict[str, Dict[str, Any]]:
    root = Path(taskspecs_dir)
    out: Dict[str, Dict[str, Any]] = {}
    for p in sorted(root.rglob("*.json")):
        try:
            raw = json.loads(p.read_text(encoding="utf-8"))
            tid = str(raw.get("task_id", "")).strip()
            if not tid:
                continue
            out[tid] = raw
        except Exception as e:
            logging.warning(f"[TASKSPEC] failed to read {p}: {e}")
    return out


def _taskspec_to_runtime(ts: Dict[str, Any]) -> TaskRuntimeSpec:
    tid = ts["task_id"]
    # global_instruction: task_text[0] 우선
    tt = ts.get("task_text", "")
    if isinstance(tt, list) and tt:
        gi = str(tt[0]).strip()
    else:
        gi = str(tt).strip()

    init_mem = ts.get("init_memory", None)
    if not isinstance(init_mem, dict):
        # 최소 안전 기본값(하지만 너는 (i)로 init_memory에 Action_Command 포함한다고 했으니 보통 여길 안 탐)
        init_mem = {
            "Working_Memory": "",
            "Episodic_Context": "",
            "Action_Command": "",
        }

    allowed = ts.get("llp_command_list", None)
    if allowed is None:
        allowed = ts.get("allowed_actions", None)
    if not isinstance(allowed, list):
        allowed = []

    return TaskRuntimeSpec(
        task_id=tid,
        global_instruction=gi,
        init_memory=init_mem,
        allowed_actions=[str(x) for x in allowed],
    )


def _make_dummy_table_image(size: Tuple[int, int] = (224, 224)) -> Image.Image:
    return Image.new("RGB", size, color=(0, 0, 0))


def eval_real_time_main_v2(
    hlp: HLPQwenV2,
    llp_cfg: LLPConfig,
    tasks: Dict[str, TaskRuntimeSpec],
    task_group_cfg: Dict[str, Any],
):
    init_llp = True
    llp_ctx: LLPRuntimeContext = init_llp_runtime(llp_cfg)

    # keyboard
    listener, kstate = init_keyboard_listener(task_group_cfg)

    # runtime state
    global_instruction: Optional[str] = None
    current_task_id: Optional[str] = None
    current_memory: Optional[Dict[str, Any]] = None

    dummy_table = _make_dummy_table_image()

    step = 0
    t0 = time.time()

    logging.info(colored("[MAIN] v2 realtime started", "green", attrs=["bold"]))
    try:
        while True:
            # ---- quit ----
            if kstate.get("quit", False):
                logging.info("[MAIN] quit")
                break

            # ---- set zero ----
            if kstate.get("set_zero", False):
                llp_send_zero(llp_ctx)
                kstate["set_zero"] = False

            # ---- reset hlp/llp (episode reset) ----
            if kstate.get("reset_hlp_llp", False):
                # episode boundary: memory is cleared and global_instruction None
                global_instruction = None
                current_task_id = None
                current_memory = None
                kstate["reset_hlp_llp"] = False
                logging.info(colored("[MAIN] reset_hlp_llp: global_instruction=None, memory cleared", "cyan"))
                time.sleep(0.05)

            # ---- task selection ----
            sel = kstate.get("selected_task", None)  # keymap value (we expect task_id)
            if sel is not None and sel != current_task_id:
                # task changed by keyboard
                prev_task_id = current_task_id
                prev_gi = global_instruction
                prev_mem = current_memory

                new_task_id = str(sel)
                if new_task_id not in tasks:
                    logging.warning(f"[MAIN] unknown task_id from keymap: {new_task_id}")
                else:
                    new_spec = tasks[new_task_id]
                    new_gi = new_spec.global_instruction

                    if prev_gi is None:
                        # None -> new task: init_memory 그대로 사용 (UPDATE 호출 X)
                        global_instruction = new_gi
                        current_task_id = new_task_id
                        current_memory = dict(new_spec.init_memory)
                        logging.info(
                            colored(
                                f"[MAIN] task set (None->new): {new_task_id} | GI='{new_gi}' | init_memory used",
                                "yellow",
                                attrs=["bold"],
                            )
                        )
                    else:
                        # prev -> new: memory는 UPDATE로만 변경
                        global_instruction = new_gi
                        current_task_id = new_task_id

                        # UPDATE 1회: (prev memory + new GI) -> new memory
                        user = build_update_user_text(
                            UPDATE_HEADER,
                            global_instruction=new_gi,
                            memory_in=prev_mem if isinstance(prev_mem, dict) else {},
                            allowed_actions=new_spec.allowed_actions,
                        )
                        # update는 dummy image
                        batch = create_hlp_update_batch(hlp.processor, dummy_table, user)
                        t_u0 = time.time()
                        new_mem = hlp.update(batch)
                        t_u = time.time() - t_u0

                        # overwrite
                        current_memory = {
                            "Working_Memory": new_mem.get("Working_Memory", ""),
                            "Episodic_Context": new_mem.get("Episodic_Context", ""),
                            "Action_Command": new_mem.get("Action_Command", ""),
                        }

                        logging.info(
                            colored(
                                f"[MAIN] task changed (prev->new): {prev_task_id} -> {new_task_id} | "
                                f"UPDATE@change t={t_u:.3f}s | Action='{current_memory.get('Action_Command','')}'",
                                "yellow",
                                attrs=["bold"],
                            )
                        )

                # consume selection
                kstate["selected_task"] = None

            # ---- if no task yet, idle ----
            if global_instruction is None or current_memory is None or current_task_id is None:
                time.sleep(0.05)
                continue

            # ---- capture shared obs once ----
            state, table_img_t, wrist_img_t = capture_shared_observation(
                piper=llp_ctx.piper,
                table_rs_cam=llp_ctx.table_rs_cam,
                wrist_rs_cam=llp_ctx.wrist_rs_cam,
                use_devices=llp_ctx.cfg.use_devices,
                use_end_pose=True,
            )
            if table_img_t is None:
                # use_devices=False는 안 쓴다고 했지만, 안전하게 idle
                time.sleep(0.05)
                continue

            # table tensor -> PIL (HLP용)
            # table_img_t shape: (H,W,3) or (3,H,W) depending on your camera wrapper.
            # 여기서는 lerobot rs_cam.image_for_inference()가 반환하는 타입에 맞게 변환해야 함.
            # 기존 코드에서는 processor에 PIL을 주는게 가장 안전하므로, 아래는 흔한 케이스 2개를 지원.
            import numpy as np
            import torch

            if isinstance(table_img_t, torch.Tensor):
                arr = table_img_t.detach().cpu().numpy()
            else:
                arr = np.array(table_img_t)

            if arr.ndim == 3 and arr.shape[0] in (1, 3):  # CHW
                arr = np.transpose(arr, (1, 2, 0))
            table_pil = Image.fromarray(arr.astype("uint8")).convert("RGB")

            # ---- HLP DETECT (real table img) ----
            user_d = build_detect_user_text(
                DETECT_HEADER,
                global_instruction=global_instruction,
                memory_in=current_memory,
            )
            b_d = create_hlp_detect_batch(hlp.processor, table_pil, user_d)

            t_h0 = time.time()
            event_detected = hlp.detect(b_d)
            t_detect = time.time() - t_h0

            # ---- HLP UPDATE only if event=true ----
            t_update = 0.0
            if event_detected:
                spec = tasks[current_task_id]
                user_u = build_update_user_text(
                    UPDATE_HEADER,
                    global_instruction=global_instruction,
                    memory_in=current_memory,
                    allowed_actions=spec.allowed_actions,
                )
                b_u = create_hlp_update_batch(hlp.processor, dummy_table, user_u)
                t_u0 = time.time()
                upd = hlp.update(b_u)
                t_update = time.time() - t_u0

                current_memory = {
                    "Working_Memory": upd.get("Working_Memory", ""),
                    "Episodic_Context": upd.get("Episodic_Context", ""),
                    "Action_Command": upd.get("Action_Command", ""),
                }

            # ---- LLP step (always uses Action_Command) ----
            cmd = str(current_memory.get("Action_Command", "")).strip()
            if not cmd:
                # 안전장치: 비어있으면 아무것도 하지 않음
                time.sleep(0.05)
                continue

            llp_batch = create_llp_batch_from_obs(
                state=state,
                table_img=table_img_t,
                wrist_img=wrist_img_t,
                task=cmd,
            )
            t_pred, t_total_llp = llp_step(llp_ctx, task_text=cmd, batch=llp_batch)

            # ---- logging ----
            step += 1
            fps = step / max(1e-6, (time.time() - t0))
            logging.info(
                colored(
                    f"[MAIN] step={step} fps={fps:.2f} task_id='{current_task_id}' "
                    f"event={event_detected} cmd='{cmd}' "
                    f"hlp_detect={t_detect:.3f}s hlp_update={t_update:.3f}s llp={t_total_llp:.3f}s",
                    "magenta",
                    attrs=["bold"],
                )
            )

            time.sleep(0.02)

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
    p.add_argument("--keymap_json", type=str, required=True, help="e.g. {'1':'task_id_a','2':'task_id_b'}")
    p.add_argument("--hlp_base", type=str, required=True)
    p.add_argument("--hlp_adapter", type=str, required=True)
    p.add_argument("--hlp_device", type=str, default="cuda:0")
    p.add_argument("--hlp_attn", type=str, default="sdpa")

    # LLP
    p.add_argument("--llp_model_path", type=str, required=True)
    p.add_argument("--dataset_repo_id", type=str, default=None)
    p.add_argument("--dataset_root", type=str, default=None)
    p.add_argument("--use_devices", action="store_true")
    p.add_argument("--no_use_devices", dest="use_devices", action="store_false")
    p.set_defaults(use_devices=True)
    p.add_argument("--llp_device", type=str, default="cuda:0")
    p.add_argument("--max_steps", type=int, default=1000000)
    args = p.parse_args()

    # load taskspecs
    raw_specs = _load_taskspecs_recursive(args.taskspecs_dir)
    tasks = {tid: _taskspec_to_runtime(ts) for tid, ts in raw_specs.items()}
    logging.info(f"[MAIN] loaded taskspecs: {len(tasks)}")

    # keymap json
    keymap = json.loads(Path(args.keymap_json).read_text(encoding="utf-8"))
    task_group_cfg = {"keymap": keymap}

    # HLP
    hlp = HLPQwenV2(
        base_model_path=args.hlp_base,
        adapter_path=args.hlp_adapter,
        device=args.hlp_device,
        attn_impl=args.hlp_attn,
        load_in_4bit=True,
    )

    # LLP cfg (필요한 DatasetConfig는 네 프로젝트 configs에 맞춰 연결해야 함)
    from configs.default import DatasetConfig
    llp_cfg = LLPConfig(
        train_dataset=DatasetConfig(repo_id=args.dataset_repo_id, root=args.dataset_root),
        policy_path=args.llp_model_path,
        use_devices=bool(args.use_devices),
        task="",
        max_steps=args.max_steps,
        device=args.llp_device,
    )

    eval_real_time_main_v2(hlp=hlp, llp_cfg=llp_cfg, tasks=tasks, task_group_cfg=task_group_cfg)