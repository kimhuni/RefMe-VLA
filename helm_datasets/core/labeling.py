from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Optional

from .spec import TaskSpec
from .templates import DEFAULT_TEMPLATE, PromptTemplate
from .data_index import DataEpisode


@dataclass(frozen=True)
class Sample:
    uid: str
    task_id: str
    chunk: str
    episode: str
    inter: int
    intra: int

    image_path: str
    input_text: str

    target_command: str
    target_progress: str

    # optional: keep raw history
    history_text: str

    # ✅ world_state는 inter에만 종속
    world_state: Optional[str]


def make_history_text(task: TaskSpec, inter: int, intra: int) -> str:
    """Create history from previous (command,progress) steps, without any physical concat."""
    lines: List[str] = []

    # Commit convention: (1,0) uses previous phase last intra as history anchor
    if task.is_commit_state(inter, intra):
        prev_inter = 0
        prev_intra = task.max_intra  # (0, n-1) 느낌: 여기서 n-1 == max_intra
        cmd = task.get_command(prev_inter, prev_intra)
        prog = task.get_progress(prev_inter, prev_intra)
        lines.append(f"- command: {cmd} | progress: {prog}")
        return "\n".join(lines)

    # Normal: use all previous intra steps within the same inter
    for i in range(0, intra):
        cmd = task.get_command(inter, i)
        prog = task.get_progress(inter, i)
        lines.append(f"- command: {cmd} | progress: {prog}")

    return "\n".join(lines)


def make_observation_text(ep: DataEpisode, image_path: Path) -> str:
    return (
        f"Image: {image_path.as_posix()}\n"
        f"Episode: {ep.chunk}/{ep.episode}\n"
        f"Event frame idx: {ep.event_frame_idx}"
    )


def make_sample(
    ep: DataEpisode,
    task: TaskSpec,
    inter: int,
    intra: int,
    variant_id: str = "0000",
    camera: str = "table",
    use_event_frame: bool = False,
    template: PromptTemplate = DEFAULT_TEMPLATE,
) -> Sample:
    # 대표 프레임 선택
    img = ep.default_image_path(use_event_frame=use_event_frame, camera=camera)

    history = make_history_text(task, inter, intra)
    obs = make_observation_text(ep, img)

    # ✅ world_state는 inter만으로 가져온다
    ws = task.get_world_state(inter)

    # input에 world_state를 넣고 싶으면 여기에 추가 가능 (지금은 메타로만 저장)
    input_text = template.render(task.task_text, history, obs)

    target_command = task.get_command(inter, intra)
    target_progress = task.get_progress(inter, intra)

    uid = f"{task.task_id}@{ep.chunk}-{ep.episode}-v{variant_id}-inter{inter}-intra{intra}"

    return Sample(
        uid=uid,
        task_id=task.task_id,
        chunk=ep.chunk,
        episode=ep.episode,
        inter=inter,
        intra=intra,
        image_path=img.as_posix(),
        input_text=input_text,
        target_command=target_command,
        target_progress=target_progress,
        history_text=history,
        world_state=ws,
    )


def sample_to_jsonl_dict(s: Sample) -> Dict[str, Any]:
    return {
        "uid": s.uid,
        "task_id": s.task_id,
        "chunk": s.chunk,
        "episode": s.episode,
        "inter": s.inter,
        "intra": s.intra,
        "image_path": s.image_path,
        "input_text": s.input_text,
        "target": {
            "command": s.target_command,
            "progress": s.target_progress,
        },
        # ✅ inter에 종속된 persistent state
        "world_state": s.world_state,
        "history_text": s.history_text,
    }