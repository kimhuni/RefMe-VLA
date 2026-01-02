# utils_keyboard.py
from __future__ import annotations
from typing import Dict, Any, Optional
from pynput import keyboard

# ====== 네가 원하는 방식: task group별 키맵 ======
TASK_KEYMAP: Dict[str, Dict[str, str]] = {
    "press_button_N_times": {
        "1": "press_blue_button_1",
        "2": "press_blue_button_2",
        "3": "press_blue_button_3",
    },
    "press_button_in_order": {
        "1": "press_BGR",
        "2": "press_BRG",
        "3": "press_GBR",
        "4": "press_GRB",
        "5": "press_RBG",
        "6": "press_RGB",
    },
    "press_button_N_times_M_times_total": {
        "1": "press_blue_button_1+1",
        "2": "press_blue_button_1+2",
        "3": "press_blue_button_1+3",
    },
    "wipe_the_window": {
        "1": "wipe_the_window",
    },
}

# ====== 고정 special key ======
# - esc: robot zero
# - 0: episode reset (HLP/LLP init 의미)
# - q: quit
# - n: next inter (task_text[0] -> task_text[1])
def init_keyboard_listener(task_group: str):
    if task_group not in TASK_KEYMAP:
        raise ValueError(f"Unknown task_group='{task_group}'. Available: {list(TASK_KEYMAP.keys())}")

    keymap = TASK_KEYMAP[task_group]

    state: Dict[str, Any] = {
        "selected_task_id": None,      # task_id
        "reset_episode": False,        # 0
        "set_zero": False,             # esc
        "quit": False,                 # q
        "next_inter": False,           # n
        "task_group": task_group,
    }

    def on_press(key):
        # esc
        if key == keyboard.Key.esc:
            state["set_zero"] = True
            print("[KEY] esc -> robot zero")
            return

        try:
            ch = key.char
        except Exception:
            ch = None

        if ch is None:
            return

        # quit
        if ch == "q":
            state["quit"] = True
            print("[KEY] q -> quit")
            return

        # reset episode
        if ch == "0":
            state["reset_episode"] = True
            print("[KEY] 0 -> reset episode (global_instruction=None, memory cleared)")
            return

        # next inter
        if ch == "`":
            state["next_inter"] = True
            print("[KEY] ` (left of key '1') -> next inter (task_text[+1])")
            return

        # numeric task select
        if ch in keymap:
            tid = keymap[ch]
            state["selected_task_id"] = tid
            print(f"[KEY] {ch} -> task_id='{tid}'")
            return

    listener = keyboard.Listener(on_press=on_press)
    listener.start()
    return listener, state