def init_keyboard_listener(task_group_cfg):
    from pynput import keyboard

    state = {
        "last_key": None,
        "reset_hlp_llp": False,
        "set_zero": False,
        "quit": False,
        "selected_task": None,
    }

    keymap = task_group_cfg["keymap"]

    def on_press(key):
        try:
            if key == keyboard.Key.esc:
                state["set_zero"] = True
                return

            if key == keyboard.KeyCode.from_char("0"):
                state["reset_hlp_llp"] = True
                return

            if key == keyboard.KeyCode.from_char("q"):
                state["quit"] = True
                return

            if hasattr(key, "char") and key.char in keymap:
                state["selected_task"] = keymap[key.char]
        except Exception:
            pass

    listener = keyboard.Listener(on_press=on_press)
    listener.start()
    return listener, state