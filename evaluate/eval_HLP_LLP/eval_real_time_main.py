# evaluate/eval_real_time_main.py
import time
import logging
from dataclasses import dataclass
from typing import Dict, Any, Optional

import argparse

from pynput import keyboard
import torch
import numpy as np
from PIL import Image

from configs.default import DatasetConfig

from evaluate.eval_HLP_LLP.eval_real_time_qwen import HLPQwen as HLPPolicy, HLP_HEADER_1, HLP_HEADER_2
from evaluate.eval_HLP_LLP.eval_real_time_pi0 import (
    LLPConfig, init_llp_runtime as init_llp, llp_step, llp_send_zero,
    capture_shared_observation, create_llp_batch_from_obs,
)
# read_end_pose_msg는 LLP 코드에서 그대로 사용 (capture_shared_observation 내부)

"""
python evaluate/eval_real_time_main.py \
  --task_group press_N_times \
  --use_devices True \
  --max_steps 1000 \
  --llp_model_path /ckpt/pi0 \
  --dataset_repo_id my_repo \
  --dataset_root /data/my_dataset \
  --llp_device cuda:0 \
  --infer_chunk 45 \
  --hlp_base_model /ckpt/Qwen2.5-VL-7B-Instruct \
  --hlp_adapter /result/ghkim/HLP_HeLM/checkpoint-2000 \
  --hlp_device cuda:0 \
  --hlp_load_in_4bit True \
  --hlp_max_new_tokens 128
"""

# =========================
# Task Group Config (main에서 관리)
# =========================
TASK_GROUPS = {
    "press_N_times": {
        "name": "Press button N times",
        "keymap": {
            "1": "press the blue button one time",
            "2": "press the blue button two times",
            "3": "press the blue button three times",
            "4": "press the blue button four times",
        },
        "prev_history": {
            "1": "Progress: 0/1 Pressed | World_State: None",
            "2": "Progress: 0/2 Pressed | World_State: None",
            "3": "Progress: 0/3 Pressed | World_State: None",
            "4": "Progress: 0/4 Pressed | World_State: None",
        },
        "default_key": "1",
    },
    "press_in_order": {
        "name": "Press buttons in order",
        "keymap": {
            "1": "press the buttons in red, green, blue order",
            "2": "press the buttons in blue, green, red order",
        },
        "default_key": "1",
    },
    "press_total": {
        "name": "Press M times total",
        "keymap": {
            "1": "press the blue button one time",
            "4": "press the blue button two times in total",
        },
        "default_key": "1",
    },
    "wipe_the_window": {
        "name": "wipe the bottom, middle, top part of the window in order",
        "keymap": {
            "1": "wipe the bottom, middle, top part of the window in order",
        },
        "prev_history": {
            "1": "nothing wiped",
        },
        "LLP_commands" : {
            "- wipe the bottom side of the window\n- wipe the middle side of the window\n- wipe the top side of the window\n- done\n"
        },
        "default_key": "1",
    },

}


@dataclass
class KeyState:
    quit: bool = False
    set_zero: bool = False
    reset_all: bool = False
    selected_task: Optional[str] = None
    prev_history: Optional[str] = None


def init_keyboard_listener(task_group_cfg: Dict[str, Any]) -> KeyState:
    st = KeyState()
    keymap = task_group_cfg["keymap"]
    prev_history = task_group_cfg["prev_history"]

    def on_press(key):
        try:
            if key == keyboard.Key.esc:
                st.set_zero = True
                print("[key] esc -> set robot zero")
                return
            if key == keyboard.KeyCode.from_char("0"):
                st.reset_all = True
                print("[key] 0 -> reset HLP/LLP")
                return
            if key == keyboard.KeyCode.from_char("q"):
                st.quit = True
                print("[key] q -> quit")
                return

            if hasattr(key, "char") and key.char in keymap:
                st.selected_task = keymap[key.char]
                st.prev_history = prev_history[key.char]
                print(f"[key] {key.char} -> task = {st.selected_task} | prev_history = {st.prev_history}")

        except Exception as e:
            print("[key] error:", e)

    listener = keyboard.Listener(on_press=on_press)
    listener.start()
    st._listener = listener  # type: ignore
    return st


def stop_keyboard_listener(st: KeyState):
    if hasattr(st, "_listener"):
        st._listener.stop()  # type: ignore

def _tensor_1chw_to_pil(img: torch.Tensor) -> Image.Image:
    """
    img: torch.Tensor, shape (1,C,H,W) or (C,H,W), float in [0,1]
    -> PIL RGB (uint8, HWC)
    """
    if img.ndim == 4:
        img = img[0]  # (C,H,W)
    img = img.detach().cpu()

    # (C,H,W) -> (H,W,C)
    img = img.permute(1, 2, 0).contiguous().numpy()

    # [0,1] -> [0,255]
    img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(img, mode="RGB")

def make_hlp_batch(processor, num_image, table_img, wrist_img, task: str, prev_memory: Optional[str], LLP_commands):
    """
    main에서 HLP 입력(batch)을 만든다.
    - 이미지 placeholder 2개 + 텍스트 프롬프트
    - Frame/Images 라인 없이 깔끔한 정책 프롬프트
    """
    prev_memory_str = prev_memory
    print("previous memory : ", prev_memory_str)

    HLP_HEADER = HLP_HEADER_1 if int(num_image) == 1 else HLP_HEADER_2

    user_text = (
        HLP_HEADER + "\n\n"
        f"Task: {task}\n"
        f"Previous_Memory: {prev_memory_str}\n"
        f"Available_LLP_Commands:\n{LLP_commands}"
        "Choose ONE command exactly as written above.\n"
    )

    if int(num_image) == 1:
        # Qwen2.5-VL style messages: 1 image + text
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": user_text},
                ],
            },
        ]

    else:
        # Qwen2.5-VL style messages: 2 images + text
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "image"},
                    {"type": "text", "text": user_text},
                ],
            },
        ]

    table_pil = _tensor_1chw_to_pil(table_img)
    if int(num_image) == 1:
        images = [table_pil]
    else:
        wrist_pil = _tensor_1chw_to_pil(wrist_img)
        images = [table_pil, wrist_pil]

    prompt = processor.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    batch = processor(
        text=[prompt],
        images=images,
        padding=True,
        return_tensors="pt",
    )
    return batch


def main(
    task_group_name: str,
    llp_cfg: LLPConfig,
    hlp_base_model: str,
    hlp_adapter: str,
    num_image: int,
    device: str = "cuda:0",
):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")

    tg = TASK_GROUPS[task_group_name]
    current_task = tg["keymap"][tg["default_key"]]
    current_history = tg["prev_history"][tg["default_key"]]
    LLP_commands = tg["LLP_commands"]
    logging.info(f"[INIT] task_group={tg['name']} default_task='{current_task}' default_history='{current_history}'")

    # --- init runtimes ---
    hlp = HLPPolicy(
        base_model_path=hlp_base_model,
        adapter_path=hlp_adapter,
        device=device,
        attn_impl="sdpa",
        load_in_4bit=True,
        max_new_tokens=128,
    )
    llp_ctx = init_llp(llp_cfg)

    # --- keyboard ---
    ks = init_keyboard_listener(tg)

    # --- loop state ---
    step_i = 0
    prev_memory: Optional[str] = None

    logging.info("[MAIN] start realtime eval loop")
    try:
        while True:
            loop_t0 = time.time()
            step_i += 1

            # -------- keyboard events --------
            if ks.quit:
                logging.info("[MAIN] quit pressed")
                break

            if ks.set_zero:
                llp_send_zero(llp_ctx)
                ks.set_zero = False

            if ks.reset_all:
                # HLP/LLP reset: memory reset + (필요하면 policy reset)
                prev_memory = None
                hlp.reset()
                # policy reset은 기존 로직을 존중 (필요하면 아래 한 줄 추가 가능)
                llp_ctx.policy.reset()
                logging.info("[MAIN] reset HLP/LLP (memory cleared)")
                ks.reset_all = False

            if ks.selected_task is not None:
                current_task = ks.selected_task
                prev_memory = ks.prev_history
                hlp.reset()
                logging.info(f"[MAIN] task changed -> '{current_task}' (memory cleared)")
                ks.selected_task = None
                ks.prev_history = None

            # -------- capture shared observation ONCE --------
            state, table_img, wrist_img = capture_shared_observation(
                piper=llp_ctx.piper,
                table_rs_cam=llp_ctx.table_rs_cam,
                wrist_rs_cam=llp_ctx.wrist_rs_cam,
                use_devices=llp_ctx.cfg.use_devices,
                use_end_pose=True,
            )
            if state is None:
                logging.warning("[MAIN] use_devices=False not supported in this realtime loop yet")
                continue



            # -------- HLP --------
            hlp_batch = make_hlp_batch(
                processor=hlp.processor,
                num_image=num_image,
                table_img=table_img,
                wrist_img=wrist_img,
                task=current_task,
                LLP_commands=LLP_commands,
                prev_memory=prev_memory,
            )

            # HLP 1 step
            t_hlp0 = time.time()
            hlp_out = hlp.forward(hlp_batch)
            hlp_t = time.time() - t_hlp0

            progress = hlp_out.get("Progress", "")
            world_state = hlp_out.get("World_State", "None")
            command = hlp_out.get("Command", "").strip()
            # command = hlp_out.get("Command", "")

            # main이 prev_memory를 관리 (overwrite)
            prev_memory = f"Progress: {progress} | World_State: {world_state}"

            # -------- termination --------
            if command == "done":
                loop_t = time.time() - loop_t0
                fps = 1.0 / max(loop_t, 1e-6)
                # logging.info(
                #     f"[MAIN] step={step_i} fps={fps:.2f} "
                #     f"task='{current_task}' cmd='done' progress='{progress}' "
                #     f"hlp_t={hlp_t:.3f}s"
                # )
                logging.info("[MAIN] HLP returned done -> stop")
                # break

            # -------- LLP (batch is created in main, uses same observation) --------
            llp_batch = create_llp_batch_from_obs(
                state=state,
                table_img=table_img,
                wrist_img=wrist_img,
                task=command,  # LLP는 HLP의 Command를 그대로 받음
            )

            t_llp0 = time.time()
            # LLP 1 step
            t_pred, t_total = llp_step(llp_ctx, task_text=command, batch=llp_batch)
            llp_t = time.time() - t_llp0

            # -------- logging --------
            loop_t = time.time() - loop_t0
            fps = 1.0 / max(loop_t, 1e-6)
            logging.info(
                f"[MAIN] step={step_i} fps={fps:.2f} "
                f"task='{current_task}' cmd='{command}' \n"
                f"progress='{progress}' world_state='{world_state}' "
                f"hlp_t={hlp_t:.3f}s llp_t={llp_t:.3f}s "
                f"(llp_pred={t_pred:.3f}s llp_total={t_total:.3f}s)"
            )

    finally:
        stop_keyboard_listener(ks)
        logging.info("[MAIN] loop finished")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")

    p = argparse.ArgumentParser()

    # Core switches
    p.add_argument("--task_group", type=str, default="press_n_times", choices=list(TASK_GROUPS.keys()))
    p.add_argument("--max_steps", type=int, default=1000)
    p.add_argument("--use_devices", type=bool, default=True)

    # LLP (pi0) config
    p.add_argument("--llp_model_path", type=str, required=True, help="Path to pi0 policy checkpoint/config")
    p.add_argument("--dataset_repo_id", type=str, required=True)
    p.add_argument("--dataset_root", type=str, required=True)
    p.add_argument("--llp_device", type=str, default="cuda:0")
    p.add_argument("--infer_chunk", type=int, default=45)

    # HLP (Qwen) config
    p.add_argument("--hlp_base_model", type=str, required=True, help="Base Qwen2.5-VL model path or repo id")
    p.add_argument("--hlp_adapter", type=str, required=True, help="Trained LoRA/QLoRA adapter path")
    p.add_argument("--num_image", type=int, default=2)
    p.add_argument("--hlp_device", type=str, default="cuda:0")
    p.add_argument("--hlp_load_in_4bit", type=bool, default=True)
    p.add_argument("--hlp_max_new_tokens", type=int, default=128)

    args = p.parse_args()

    # Build LLPConfig (minimal required fields)
    llp_cfg = LLPConfig(
        train_dataset=DatasetConfig(
        repo_id=args.dataset_repo_id,
        root=args.dataset_root,
        ),
        policy_path=args.llp_model_path,
        use_devices=bool(args.use_devices),
        task="",
        max_steps=args.max_steps,
        device=args.llp_device,
        infer_chunk=args.infer_chunk,
    )

    # Call main loop
    main(
        task_group_name=args.task_group,
        llp_cfg=llp_cfg,
        hlp_base_model=args.hlp_base_model,
        hlp_adapter=args.hlp_adapter,
        num_image=args.num_image,
        device=args.hlp_device,
    )