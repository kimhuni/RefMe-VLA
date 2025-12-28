def create_hlp_batch(processor, table_img, wrist_img, task, previous_memory):
    prompt = (
        "Role: High-Level Planner (HLP).\n"
        "Given the two images and Previous_Memory, update the memory and choose the next atomic command.\n"
        "- Only advance Progress when the event has occurred in the current frame.\n"
        "- World_State should be concise and persistent (use None if no state).\n"
        "- Command should be either the task command or \"done\" if finished.\n"
        "Return YAML with keys Progress, World_State, Command.\n\n"
        f"Task: {task}\n"
        f"Previous_Memory: {previous_memory if previous_memory else 'None'}"
    )

    return processor(
        text=[prompt],
        images=[table_img, wrist_img],
        padding=True,
        return_tensors="pt",
    )


def create_llp_batch(processor, table_img, wrist_img, command):
    return {
        "images": (table_img, wrist_img),
        "command": command,
    }