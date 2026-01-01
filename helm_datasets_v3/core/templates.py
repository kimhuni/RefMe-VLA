from __future__ import annotations
import yaml
from typing import Dict, Optional


# v3: output은 YAML (학습 대상)
def dump_yaml(d: Dict) -> str:
    # 사람이 읽기 쉬운 YAML, 키 순서 유지
    return yaml.safe_dump(d, sort_keys=False, allow_unicode=True).strip()


DETECT_SYSTEM = (
    "You are the robot arm Visual Event Detector.\n"
    "Goal: Decide whether the target EVENT is detected in the current image.\n"
    "The EVENT corresponds to a meaningful completion moment for the current stage of the Global_Instruction."
    "Input: An image + Global_Instruction describing what counts as action completion"
    " + Memory (may help interpret the current stage/goal)\n"
    "Decision rule:\n"
    "- Use the Global_Instruction  as the primary criterion.\n"
    "- You MAY use Memory only to understand what “completion” means for the current stage."
    "- Event_Detected: true when the completion (or clearly post-completion state) is visible.\n"
    "- Otherwise (partial progress / occlusion / uncertainty) -> Event_Detected: false.\n"
    "Constraints:\n"
    "- Do not propose next actions.\n"
    "- Do not update or rewrite memory.\n"
    "- Do not output any text except YAML.\n"
    "Return YAML with exactly one key: Event_Detected (boolean).\n"
)

UPDATE_SYSTEM = (
    "You are the robot arm Logic State Manager.\n"
    "Context: Event_Detected=true or a Task Change has occurred.\n"
    "Inputs:\n"
    "- Global_Instruction defining the overall task.\n"
    "- Previous memory state (with keys: Working_Memory, Episodic_Context, Action_Command).\n"
    "- Allowed_Action_Commands (a small fixed list)"
    "Goal: Produce the next memory state after the event, preserving information"
    "and decide the next Action_Command based on the Global_Instruction.\n"
    "Logic Rules ((copy-first, lossless)):\n"
    "1) Start by COPYING Previous_Memory fields.\n"
    "2) Update Working_Memory to reflect the newly completed step."
    "- Prefer appending or small edits over rewriting."
    "3) Episodic_Context:"
    "- If the task is not finished, keep it EXACTLY unchanged."
    "- If the task is finished, update it to summarize the final outcome."
    "4) Action_Command:"
    "- Must be EXACTLY one of Allowed_Action_Commands."
    "- Use done only when the task is finished."
    "Constraints:\n"
    "- Action_Command must be selected ONLY from Allowed_Action_Commands.\n"
    "- Do not add new actions or explanations.\n"
    "- Output YAML only with keys: Action_Command, Working_Memory, Episodic_Context.\n"
)

def render_memory_one_line(mem: Dict[str, str]) -> str:
    # prompt 입력은 한 줄로 짧게, 출력은 YAML로 강제
    ac = mem.get("Action_Command", "None")
    wm = mem.get("Working_Memory", "None")
    ec = mem.get("Episodic_Context", "None")
    return f"Action_Command: {ac} | Working_Memory: {wm} | Episodic_Context: {ec}"


def make_detect_prompt(task_text: str, memory: Dict[str, str], n_images: int) -> str:
    img_tokens = "<image_table>" + (" <image_wrist>" if n_images == 2 else "")
    return (
        f"{DETECT_SYSTEM}\n"
        f"Task: {task_text}\n"
        f"Memory: {render_memory_one_line(memory)}\n"
        f"Images: {img_tokens}\n"
    )



def make_update_prompt(
    task_text: str,
    prev_memory: Dict[str, str],
    n_images: int,
    llp_commands: str
) -> str:
    img_tokens = "<image_table>" + (" <image_wrist>" if n_images == 2 else "")
    allowed_block = ""
    if llp_commands.strip():
        allowed_block = f"\nAllowed_Action_Commands:\n{llp_commands.strip()}\n"

    return (
        f"{UPDATE_SYSTEM}\n"
        f"Task: {task_text}\n"
        f"Previous_Memory: {render_memory_one_line(prev_memory)}\n"
        f"{allowed_block}"
        f"Images: {img_tokens}\n"
    )