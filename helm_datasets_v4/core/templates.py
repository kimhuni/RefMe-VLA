from __future__ import annotations
import yaml
from typing import Dict, Optional


# v4: output은 YAML (학습 대상)
def dump_yaml(d: Dict) -> str:
    # 사람이 읽기 쉬운 YAML, 키 순서 유지
    return yaml.safe_dump(d, sort_keys=False, allow_unicode=True).strip()

# v4
DETECT_SYSTEM = (
    "Role: Robot arm Visual Event Detector (DETECT mode).\n"
    "Goal: Decide whether a stage-completion EVENT is visible in the current image, and output the event name.\n\n"

    "Inputs:\n"
    "- Task (Global_Instruction): the current stage instruction.\n"
    "- Event_List: newline-separated list of allowed event strings.\n"
    "- Memory: {Action_Command, Working_Memory, Episodic_Context} (use only to interpret the current stage).\n"
    "- Images.\n\n"

    "Decision rules (be conservative):\n"
    "1) Use Task (Global_Instruction) as the PRIMARY criterion.\n"
    "2) You MAY use Memory only to clarify what counts as completion for the current stage.\n"
    "3) Set Event_Detected=true ONLY when the completion state (or clearly post-completion state) is visible in image.\n"
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

# v4
UPDATE_SYSTEM = (
    "Role: Robot arm Logic State Manager (UPDATE mode).\n"
    "When: Event happened (from DETECT) OR Task changed (new stage).\n\n"

    "Inputs:\n"
    "- Task: current instruction.\n"
    "- Event: detected event string (or 'none').\n"
    "- Previous_Memory: {Action_Command, Working_Memory, Episodic_Context}.\n"
    "- Allowed_Action_Commands: you MUST output EXACTLY one of them.\n\n"

    "Goal:\n"
    "Produce the NEXT memory state (copy-first) and choose the next Action_Command.\n\n"

    "Three cases:\n"
    "1) INIT MEMORY (episode start or new task start):\n"
    "   - If Previous_Memory is empty/None, initialize memory from Task.\n"
    "   - Set Working_Memory using a simple structured format inferred from Task "
    "(e.g., 'Count: X (Goal: Y)' or 'Progress: P (Goal: G)').\n\n"
    "2) WORKING MEMORY UPDATE (task not finished):\n"
    "   - Update Working_Memory ONLY as needed for this Event.\n"
    "   - Keep Episodic_Context EXACTLY unchanged.\n"
    "   - Choose the next Action_Command (not 'done').\n\n"
    "3) DONE (task finished after this update):\n"
    "   - Action_Command MUST be 'done'.\n"
    "   - Working_Memory MUST be 'task done (None)'.\n"
    "   - Update Episodic_Context with a short structured summary consistent with the memory style "
    "(e.g., 'Previous_Count: K' or 'Previous_Progress: ...').\n\n"

    "Constraints:\n"
    "- Do not invent new actions, keys, or free-form formats; follow the dataset style.\n"
    "Output YAML only with keys: Action_Command, Working_Memory, Episodic_Context.\n"
)

def render_memory_one_line(mem: Dict[str, str]) -> str:
    # prompt 입력은 한 줄로 짧게, 출력은 YAML로 강제
    ac = mem.get("Action_Command", "None")
    wm = mem.get("Working_Memory", "None")
    ec = mem.get("Episodic_Context", "None")
    return f"Action_Command: {ac} | Working_Memory: {wm} | Episodic_Context: {ec}"


def make_detect_prompt(
    task_text: str,
    memory: Dict[str, str],
    n_images: int,
    event_list: Optional[str] = None,
) -> str:
    img_tokens = "<image_table>" + (" <image_wrist>" if n_images == 2 else "")

    event_block = ""
    if event_list is not None:
        # newline-separated, like llp_commands
        event_block = f"Event_List:\n{event_list}\n"

    return (
        f"{DETECT_SYSTEM}\n"
        f"Task: {task_text}\n"
        f"{event_block}"
        f"Memory: {render_memory_one_line(memory)}\n"
        f"Images: {img_tokens}\n"
    )


def make_update_prompt(
    task_text: str,
    prev_memory: Dict[str, str],
    n_images: int,
    llp_commands: str,
    event: Optional[str] = None,
) -> str:
    img_tokens = "<image_table>" + (" <image_wrist>" if n_images == 2 else "")

    allowed_block = ""
    if llp_commands.strip():
        allowed_block = f"\nAllowed_Action_Commands:\n{llp_commands.strip()}\n"

    event_line = ""
    if event is not None:
        event_line = f"Event: {event}\n"

    return (
        f"{UPDATE_SYSTEM}\n"
        f"Task: {task_text}\n"
        f"{event_line}"
        f"Previous_Memory: {render_memory_one_line(prev_memory)}\n"
        f"{allowed_block}"
        f"Images: {img_tokens}\n"
    )