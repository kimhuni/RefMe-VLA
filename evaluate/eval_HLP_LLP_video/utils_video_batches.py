# utils_video_batches.py
from __future__ import annotations
from typing import Any, Dict, List, Optional
import yaml

# --- Video Baseline Templates (From templates_visual.py) ---

DETECT_SYSTEM = (
    "Role: Robot arm Visual Event Detector (DETECT mode).\n"
    "Goal: Decide whether a stage-completion EVENT is visible in the current image, and output the event name.\n\n"
    "Inputs:\n"
    "- Task (Global_Instruction): the current stage instruction.\n"
    "- Event_List: newline-separated list of allowed event strings.\n"
    "- Action_Command: The current action being executed.\n"
    "- Images.\n\n"
    "Decision rules (be conservative):\n"
    "1) Use Task (Global_Instruction) as the PRIMARY criterion.\n"
    "2) Set Event_Detected=true ONLY when the completion state (or clearly post-completion state) is visible in image.\n"
    "   - If partial progress / occlusion / uncertainty -> Event_Detected=false.\n\n"
    "Event selection rules:\n"
    "- If Event_Detected=false: Event MUST be exactly 'none'.\n"
    "- If Event_Detected=true: Event MUST be EXACTLY one item from Event_List.\n"
    "- Do NOT invent new events. Use EXACT string match.\n\n"
    "Constraints:\n"
    "- Do not propose next actions.\n"
    "- Do not update or rewrite memory.\n"
    "- Output YAML only.\n\n"
    "Output YAML with EXACTLY these keys:\n"
    "- Event_Detected: boolean\n"
    "- Event: string\n"
)

# UPDATE_SYSTEM = (
#     "Role: Robot arm Logic State Manager (UPDATE mode).\n"
#     "When: Event happened (from DETECT) OR Task changed (new stage).\n\n"
#     "Inputs:\n"
#     "- Task: current instruction.\n"
#     "- Event: detected event string (or 'none').\n"
#     "- Previous_Action_Command: The action that was just completed.\n"
#     "- Allowed_Action_Commands: you MUST output EXACTLY one of them.\n"
#     "- Images: A sequence of historical event frames showing the progress so far.\n\n"
#     "Goal:\n"
#     "Analyze the visual history and the current event to choose the next Action_Command.\n\n"
#     "Constraints:\n"
#     "- Do not invent new actions, keys, or free-form formats; follow the dataset style.\n"
#     "- Action_Command MUST be 'done' if the task is finished.\n"
#     "Output YAML only with keys: Action_Command\n"
# )

UPDATE_SYSTEM = (
    "Role: Robot arm Logic State Manager (UPDATE mode).\n"
    "When: Event happened (from DETECT) OR Task changed (new stage).\n\n"
    "Inputs:\n"
    "- Task: current instruction.\n"
    "- Event: detected event string.\n"
    "- Previous_Action_Command: The action that was just completed.\n"
    "- Allowed_Action_Commands: you MUST output EXACTLY one of them.\n"
    "- Images: A sequence of historical event frames showing the progress so far.\n\n"
    "Goal:\n"
    "Carefully analyze the visual history (Images) to verify which sub-tasks have been completed.\n"
    "Compare the visual evidence against the 'Task' requirements to choose the next Action_Command.\n\n"

    "Decision Rules (CRITICAL):\n"
    "1. Scan the 'Images' sequence to list all actions completed so far.\n"
    "2. Compare completed actions with the required steps in 'Task'.\n"
    "3. If ANY step is missing in the visual history, output the action for that missing step.\n"
    "4. Output 'done' ONLY IF the visual history proves ALL required steps are finished.\n"
    "   - Do NOT output 'done' if there are remaining steps.\n\n"

    "Constraints:\n"
    "- Do not invent new actions, keys, or free-form formats; follow the dataset style.\n"
    "- Action_Command MUST be 'done' if the task is finished.\n"
    "Output YAML only with keys: Action_Command\n"
)


def make_video_detect_prompt(
        task_text: str,
        action_command: str,
        event_list: str,
) -> str:
    return (
        f"{DETECT_SYSTEM}\n"
        f"Task: {task_text}\n"
        f"Event_List:\n{event_list}\n"
        f"Action_Command: {action_command}\n"
        f"Images: <image_table>\n"
    )


def make_video_update_prompt(
        task_text: str,
        prev_action: str,
        event: str,
        allowed_commands: str,
) -> str:
    return (
        f"{UPDATE_SYSTEM}\n"
        f"Task: {task_text}\n"
        f"Event: {event}\n"
        f"Previous_Action_Command: {prev_action}\n"
        f"Allowed_Action_Commands:\n{allowed_commands}\n"
        f"Images: <image_table>\n"
    )


# --- Batch Creation ---

def create_video_detect_batch(processor, obs_pil, user_text: str):
    """
    DETECT: Single Image Input
    """
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": user_text}
            ]
        }
    ]
    prompt = processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return processor(text=prompt, images=obs_pil, padding=True, return_tensors="pt")


def create_video_update_batch(processor, history_images: List[Any], user_text: str):
    """
    UPDATE: Multi-Image Input (Visual History)
    """
    # Create N image placeholders
    content = [{"type": "image"} for _ in range(len(history_images))]
    content.append({"type": "text", "text": user_text})

    messages = [{"role": "user", "content": content}]

    prompt = processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return processor(text=prompt, images=history_images, padding=True, return_tensors="pt")