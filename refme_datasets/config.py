# datasets/subtask/config.py

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class VelocityLabelConfig:
    """
    속도 기반 subtask 라벨링을 위한 하이퍼파라미터 집합.
    """
    # 속도 임계값 (저속 판단)
    tau_low: float = 2000

    # 처음 몇 초는 boundary 후보에서 제외 (초기 노이즈/출발 구간 무시)
    ignore_first_sec: float = 1.0

    # 서로 다른 subtask로 인정하기 위한 최소 간격 (초 단위)
    min_gap_sec: float = 0.4

    # 샘플링 주파수 (Hz). timestamp가 있으면 그걸 우선 사용.
    freq_hz: float = 5.0

    # 저속 구간 길이가 이 값(초)보다 짧으면 노이즈로 보고 버릴 수도 있음
    min_segment_sec: float = 0.2


# task_index → subtask 개수 매핑
# (실제 프로젝트에서 task_index / task 문자열에 맞게 조정하면 됨)
TASK_INDEX_TO_NUM_SUBTASKS: dict[int, int] = {
    0: 3,   # press the blue button
    1: 6,   # press the blue button two times
    2: 9,   # press the blue button three times
    3: 12,  # press the blue button four times
    4: 6,   # press red, green, blue in order
}

tasks_dict = {
    # '0' : "press the blue button",
    '1': "press the blue button two times",
    '2': "press the blue button three times",
    '3': "press the blue button four times",
    # '4' : "press the red, green, blue buttons in order",
}