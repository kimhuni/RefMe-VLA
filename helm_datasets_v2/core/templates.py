# helm_datasets/core/templates.py
from __future__ import annotations

DETECT_SYSTEM = (
    "You are HeLM High-Level Planner in DETECT mode.\n"
    "Given the current images and Previous_Memory:\n"
    "- Decide whether the task-specific event has occurred in the CURRENT frame.\n"
    "- Choose the next command for the low-level policy (LLP).\n"
    "Rules:\n"
    "- Event_Detected must be true only if the event is visible in the current frame.\n"
    '- If the task is finished, Command must be "done".\n'
    '- Command must be either the given LLP_Command or "done".\n'
    "Return YAML with keys: Event_Detected, Command.\n"
)

UPDATE_SYSTEM = (
    "You are HeLM Memory Updater in UPDATE mode.\n"
    "Event_Detected is TRUE for the current frame.\n"
    "Update the memory based on Previous_Memory and current command which is done.\n"
    "Rules:\n"
    "- Update Progress by current command which is just done. Remember the previous Progress.\n"
    "- Keep World_State concise and persistent. The World_State is updated only when task is changed.\n"
    "Return YAML with keys: Progress, World_State.\n"
)


def make_prev_memory(progress: str, world_state: str) -> str:
    ws = world_state if (world_state and str(world_state).strip()) else "None"
    return f"Progress: {progress} | World_State: {ws}"


def render_user_prompt_detect(task_text: str, llp_command: str, prev_memory: str) -> str:
    return (
        f"{DETECT_SYSTEM}\n\n"
        "MODE: DETECT\n"
        f"Task: {task_text}\n"
        f"LLP_Command: {llp_command}\n"
        f"Previous_Memory: {prev_memory}\n"
    )


def render_user_prompt_update(task_text: str, prev_memory: str) -> str:
    return (
        f"{UPDATE_SYSTEM}\n\n"
        "MODE: UPDATE\n"
        f"Task: {task_text}\n"
        f"Previous_Memory: {prev_memory}\n"
        "Event_Detected: true\n"
    )


def render_assistant_yaml_detect(event_detected: bool, command: str) -> str:
    ev = "true" if event_detected else "false"
    return f"Event_Detected: {ev}\nCommand: {command}\n"


def render_assistant_yaml_update(progress: str, world_state: str) -> str:
    ws = world_state if (world_state and str(world_state).strip()) else "None"
    return f"Progress: {progress}\nWorld_State: {ws}\n"
