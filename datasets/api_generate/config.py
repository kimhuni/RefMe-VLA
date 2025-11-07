# config.py
from dataclasses import dataclass
from pathlib import Path
from typing import List

STATUS_SET = {"DONE", "NOT_DONE", "UNCERTAIN"}
######## ---- Prompt registry -------- Prompt registry -------- Prompt registry -------- Prompt registry ----###########
# 키: prompt_id, 값: 포맷 함수(task:str, prev:str) -> str

# prompt 1
def _prompt_B_eval_v3(task: str, prev: str) -> str:
    return (
        f"TASK: {task}\n"
        f"PREV: {prev}\n"
        "INSTRUCTIONS:\n"
        "- Only describe visible evidence (contact, alignment, illumination).\n"
        "- Exactly TWO sentences (<=12 words each), then one of {DONE, NOT_DONE, UNCERTAIN}.\n"
        "- No speculation beyond the frame.\n"
        "OUTPUT:\n"
        "Return only ONE LINE minified JSON: "
        "{\"desc_1\":\"...\",\"desc_2\":\"...\",\"status\":\"DONE|NOT_DONE|UNCERTAIN\"}"
    )

# prompt 2
def _prompt_B_eval_compact(task: str, prev: str) -> str:
    return (
        f"TASK:{task}\nPREV:{prev}\n"
        "Rules: 2 short sentences about visible evidence only. Then DONE/NOT_DONE/UNCERTAIN.\n"
        "JSON one line: {\"desc_1\":\"...\",\"desc_2\":\"...\",\"status\":\"DONE|NOT_DONE|UNCERTAIN\"}"
    )

# prompt 3
def _prompt_B_eval_strict_json(task: str, prev: str) -> str:
    return (
        f"TASK:{task}\nPREV:{prev}\n"
        "Return ONLY this JSON (no extra text): "
        "{\"desc_1\":\"...\",\"desc_2\":\"...\",\"status\":\"DONE|NOT_DONE|UNCERTAIN\"}"
    )

# ---- New prompts (based on user's original system prompt) ----
def _prompt_B_eval_user_mini(task: str, prev: str) -> str:
    # 초간소: 규칙 핵심 + JSON 한 줄
    return (
        f"TASK: {task}\n"
        f"PREV: {prev}\n"
        "You are an image analysis expert for robot manipulation.\n"
        "Write EXACTLY two short sentence from the CURRENT image only.\n"
        "Use evidence-based completion: mark DONE only with clear visible contact/result; otherwise NOT_DONE or UNCERTAIN.\n"
        "Do not rely on PREV as proof; each frame must stand on its own.\n"
        "Return ONE LINE JSON(no extra text): {\"desc_1\":\"...\",\"desc_2\":\"...\",\"status\":\"DONE|NOT_DONE|UNCERTAIN\"}"
    )

def _prompt_B_eval_user_medium(task: str, prev: str) -> str:
    # 적당 요약: 원본의 구조/규칙을 간결히 유지
    return (
        "You are an image analysis expert specialized in robotic manipulation.\n"
        f"TASK: {task}\n"
        f"PREV: {prev}\n"
        "GOAL:\n"
        "- S1: Describe what the robot arm is doing in the CURRENT image.\n"
        "- S2: Judge DONE/NOT_DONE/UNCERTAIN using visible evidence + task logic.\n"
        "RULES:\n"
        "- Each frame must stand on its own; PREV is for continuity only.\n"
        "- Evidence-based completion:\n"
        "  * Press: visible contact/compression/deformation.\n"
        "  * Place: object released and stationary at target.\n"
        "  * Grasp: gripper closed with object secured.\n"
        "- Be conservative: if contact/result isn’t clear → NOT_DONE; if view/lighting blocks verification → UNCERTAIN.\n"
        "- Use concise visual verbs (moving, grasping, pressing, placing, releasing, aligning...).\n"
        "OUTPUT (one line JSON): {\"desc_1\":\"...\",\"desc_2\":\"...\",\"status\":\"DONE|NOT_DONE|UNCERTAIN\"}"
    )

def _prompt_B_eval_user_long(task: str, prev: str) -> str:
    # 원본에서 군더더기만 조금 정리한 버전(가독성 유지)
    return (
        "You are an image analysis expert specialized in robotic manipulation.\n"
        "You are given: (1) a task description, (2) a previous step output (two sentences: last-frame action + done-judgment), "
        "(3) a pair current image showing the new state. \n"
        "Your job:\n"
        "- Describe what the robot arm is doing in the CURRENT image.\n"
        "- Determine whether the task is done, considering the task and the previous output.\n"
        "Rules (critical):\n"
        "1) Write exactly two sentences:\n"
        "   - Sentence 1: Describe the visible action/attempt in the current image.\n"
        "   - Sentence 2: Judge DONE/NOT_DONE/UNCERTAIN strictly from visible evidence and task logic.\n"
        "2) Use the previous output only for continuity, not as proof of progress. Each frame stands on its own visual evidence.\n"
        "3) Evidence-based completion:\n"
        "   - Press: visible contact/compression/deformation of the button.\n"
        "   - Place: object released and stationary at target.\n"
        "   - Grasp: gripper closed with object securely held.\n"
        "   If not clearly visible, mark NOT_DONE or UNCERTAIN accordingly.\n"
        "4) Be conservative: any gap/no deformation/unclear contact → NOT_DONE. Obscured by angle/lighting → UNCERTAIN.\n"
        "5) Use concise visual verbs (moving, grasping, lifting, placing, releasing, pressing, returning, hovering, aligning).\n"
        "Template:\n"
        f"Task: {task}\n"
        f"Previous output: {prev}\n"
        "Instruction: Describe in two sentences and judge completion per the rules.\n"
        "Return ONE LINE JSON(no extra text): {\"desc_1\":\"...\",\"desc_2\":\"...\",\"status\":\"DONE|NOT_DONE|UNCERTAIN\"}"
    )

def _prompt_eval(task: str, prev: str) -> str:
    # 원본에서 군더더기만 조금 정리한 버전(가독성 유지)
    return (
        "You are an image analysis expert specialized in robotic manipulation.\n"
        f"TASK: {task}\n"
        f"PREV_JSON: {prev}\n"
        "GOAL:\n"
        "- S1 (scene): Describe what is VISIBLY present/occurring in the CURRENT image.\n"
        "- S2 (reason): State the KEY visual evidence for completion or lack thereof.\n"
        "- Status: DONE / NOT_DONE.\n"
        "PRINCIPLES:\n"
        "- Evidence-first: use ONLY the CURRENT frame as proof and use PREV if not certain\n"
        "- PREV is allowed for continuity to preserve stable facts if the scene did not change.\n"
        "CONSISTENCY (Flip-Guard):\n"
        "- Keep the prior status unless the CURRENT image clearly contradicts it.\n"
        "- If PREV.status == DONE, keep DONE unless visible undo exists.\n"
        "- If PREV.status == NOT_DONE, switch to DONE only with decisive cues.\n"
        "- If the frame shows minimal/no change, keep status and carry forward consistent scene details.\n"
        "EVIDENCE BY TASK (examples):\n"
        "- Press: contact + button depression/indicator change.\n"
        "AMBIGUITY:\n"
        "- If occlusion/blur prevents verification, set UNCERTAIN, BUT still describe what IS visible\n"
        "  and PRESERVE information from the previous description by description\n"
        " (e.g., relative positions, openness of gripper, presence/absence of contact).\n"
        "STYLE & CONSTRAINTS:\n"
        "- desc_1 and desc_2 MUST be not empty, and each 8–20 words.\n"
        "- Describe only visible facts; do not invent hidden details.\n"
        "- You may paraphrase stable facts from PREV for continuity IF the scene appears unchanged.\n"
        "- Forbidden for each output keys: empty strings, 'N/A', 'None'.\n"
        "OUTPUT (ONE LINE JSON only): "
        "{\"desc_1\":\"...\",\"desc_2\":\"...\",\"status\":\"DONE|NOT_DONE\"}\n"
    )

def _prompt_eval_short(task: str, prev: str) -> str:
    return (
        "You are an image analysis expert for robot manipulation.\n"
        f"TASK: {task}\n"
        f"PREV_JSON: {prev}\n"
        "VIEWS:\n"
        "- [TABLE]=global layout. - [WRIST]=close-up contact check. WRIST overrides TABLE for contact/depression.\n"
        "GOAL:\n"
        "- scene: CURRENT frame visible action/state. - reason: key visual evidence for completion/lack. - status: DONE/NOT_DONE/UNCERTAIN.\n"
        "RULES:\n"
        "- Evidence-first: use CURRENT frame only as proof. PREV is for continuity phrasing (not as proof) when the scene is unchanged.\n"
        "- Flip-Guard: keep prior status unless CURRENT frame clearly contradicts it. Minimal change ⇒ keep status.\n"
        "- If view is occluded/blurred, set UNCERTAIN but still describe what IS visible and which VIEW you relied on.\n"
        "CONSTRAINTS:\n"
        "- desc_1 and desc_2 MUST be non-empty (8–25 words each). Do not output empty strings, N/A, or None.\n"
        "- Mention VIEW used in reason: e.g., \"[WRIST] shows no contact\" or \"[TABLE] shows object still away\".\n"
        "OUTPUT (one line JSON only): "
        "{\"desc_1\":\"...\",\"desc_2\":\"...\",\"status\":\"DONE|NOT_DONE|UNCERTAIN\"}"
    )

def _prompt_eval_mix(task: str, prev: str) -> str:
    return (
        "You are an image analysis expert specialized in robotic manipulation.\n"
        "You are given: (1) a task description, (2) a previous step output(done-judgment), "
        "(3) a pair current image showing the new state. \n"
        f"TASK: {task}\n"
        f"PREV: {prev}\n"
        "VIEWS:\n"
        "- [TABLE]: global scene view (object layout, task goal, environment overview).\n"
        "- [WRIST]: close-up wrist camera (contact, grasp details). WRIST overrides TABLE for contact evidence.\n"
        "GOAL:\n"
        "- desc_1 (scene): Describe what is VISIBLY happening in the CURRENT frame.\n"
        "- desc_2 (reason): State the KEY visual evidence for task completion or lack thereof.\n"
        "- status: DONE / NOT_DONE / UNCERTAIN.\n"
        "PRINCIPLES:\n"
        "- Evidence-first: use ONLY the CURRENT frame(s) as proof. PREV is for continuity phrasing (not as evidence) when the scene is unchanged.\n"
        "- Flip-Guard: maintain the prior status unless the CURRENT frame clearly contradicts it.\n"
        "- If PREV.status == DONE, keep DONE unless visible undo exists.\n"
        "- If PREV.status == NOT_DONE, switch to DONE only with decisive cues.\n"
        "- If PREV.status == UNCERTAIN and evidence is still weak, prefer NOT_DONE.\n"
        "- If the frame shows minimal/no change, keep status and carry forward consistent scene details.\n"
        "EVIDENCE BY TASK (examples):\n"
        "- Press: visible gripper/button contact + indicator change.\n"
        "AMBIGUITY HANDLING:\n"
        "- If occlusion, blur prevents verification, set UNCERTAIN but still describe what IS visible\n"
        "  (e.g., relative positions, openness of gripper, contact visibility) and mention which VIEW you relied on ([TABLE] or [WRIST]).\n"
        "STYLE & HARD CONSTRAINTS:\n"
        "- desc_1 and desc_2 MUST each contain 8–15 words and MUST NOT be empty under any circumstance.\n"
        "- Absolutely forbidden outputs for any key: empty string, whitespace-only string, 'N/A', 'None', or null value.\n"
        "- If unsure, describe what is visibly unchanged or ambiguous; never leave desc fields blank.\n"
        "- Use concise visual verbs (hovering, pressing, holding, releasing) and reference VIEW when relevant.\n"
        "OUTPUT (ONE LINE JSON only, no line breaks, no extra text): "
        "{\"desc_1\":\"...\",\"desc_2\":\"...\",\"status\":\"DONE|NOT_DONE|UNCERTAIN\"}"
    )

def _prompt_final(task: str, prev: str, prev_status: str) -> str:
    return (
        "You are an image-analysis expert for robotic manipulation.\n"
        "You are given: (1) a task description, (2) a previous step output and done-judgment, "
        "(3) a pair current image showing the new state. \n"
        f"(1) TASK: {task}\n"
        f"(2) previous_output: {prev}\n"
        f"(3) previous_status: {prev_status}\n"
        "IMAGES: [TABLE]=global scene view; [WRIST]=close-up wrist camera (contact, grasp details)\n"
        "GOAL\n"
        "- desc_1=what is VISIBLY happening now\n"
        "- desc_2=key visible evidence for task completion\n"
        "- status=based on previous output and desc_2, only choose from DONE/NOT_DONE/UNCERTAIN.\n"
        "DECISION POLICY:\n"
        "- change from NOT_DONE to DONE if success cue(s) visible on BOTH VIEWs"
        "- either any physical change OR other distinct visual feedback indicating goal achieved (e.g.,  illumination, contact, state change, release).\n"
        "- DONE: must keep DONE state from previous_status \n"
        "- NOT_DONE: visible counter-evidence that goal is not yet met. \n"
        "- UNCERTAIN: result not verifiable, or view/blur/occlusion limits. avoid only when sufficient evidence exists. \n"
        "- If the frame shows minimal/no change, keep status and carry forward consistent scene details.\n"
        "CONSISTENCY (Flip-Guard): keep prior status unless CURRENT contradicts it.\n"
        "STYLE & CONSTRAINTS:\n"
        "- desc_1 and desc_2 each 8–12 words, non-empty; mention VIEW when relevant; no invented details.\n"
        "- Forbidden: empty/whitespace, 'N/A', 'None', null.\n"
        "OUTPUT (one-line JSON only): "
        "{\"desc_1\":\"...\",\"desc_2\":\"...\",\"status\":\"DONE|NOT_DONE|UNCERTAIN\"}"
    )

PROMPTS = {
    "B_eval_v3": _prompt_B_eval_v3,
    "B_eval_compact": _prompt_B_eval_compact,
    "B_eval_strict_json": _prompt_B_eval_strict_json,
    "B_eval_user_mini": _prompt_B_eval_user_mini,
    "B_eval_user_medium": _prompt_B_eval_user_medium,
    "B_eval_user_long": _prompt_B_eval_user_long,
    "eval_1031":  _prompt_eval,
    "eval_1031_short": _prompt_eval_short,
    "eval_1031_mix": _prompt_eval_mix,
    "eval_final": _prompt_final,
}

########################################################################################################################

def render_prompt(prompt_id: str, task: str, prev_desc: str, prev_status: str) -> str:
    if prompt_id not in PROMPTS:
        keys = ", ".join(sorted(PROMPTS.keys()))
        raise ValueError(f"Unknown prompt_id '{prompt_id}'. Available: {keys}")
    print("prev_desc : ", prev_desc)
    print("prev_status : ", prev_status)

    return PROMPTS[prompt_id](task, prev_desc, prev_status)

@dataclass
class Paths:
    dataset_root: Path                 # /data/piper_push/lerobot_5hz
    derived_root: Path                 # /data/piper_push_derived

    @property
    def videos(self) -> Path:
        return self.dataset_root / "videos"

    @property
    def episodes_meta(self) -> Path:
        return self.dataset_root / "meta" / "episodes.jsonl"

    @property
    def frames_root(self) -> Path:
        return self.derived_root / "frames_1fps"

    def out_root(self, prompt_id: str) -> Path:
        return self.derived_root / prompt_id

    @property
    def shards_dir(self) -> Path:
        return self.out_root / "shards"

    @property
    def failures_file(self) -> Path:
        return self.out_root / "failures.jsonl"

    @property
    def index_file(self) -> Path:
        return self.out_root / "dataset.index"

    @property
    def stats_dir(self) -> Path:
        return self.out_root / "stats"

@dataclass
class GenConfig:
    fps: int = 1
    provider: str = "gemini"          # "openai" 또는 커스텀 제공자 키
    model: str = "gemini-1.5-pro"
    temperature: float = 0.2
    top_p: float = 0.9
    max_tokens: int = 384
    seed: int = 1234
    prompt_id: str = "B_eval_v3"
    cameras: List[str] = ("side", "wrist")
    # 프롬프트 길이 최적화
    prev_desc_max_chars: int = 160
