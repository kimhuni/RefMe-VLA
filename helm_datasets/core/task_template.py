from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Any

@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    task_text: str
    max_intra_counter: int  # 예: 3 (0,1,2,3 총 4상태)
    command_list: List[str]  # 길이 = max_intra_counter + 1
    progress_list: List[str]  # 길이 = max_intra_counter + 1
    world_state_list: List[Optional[dict]]  # 길이 = max_intra_counter + 1

    def validate(self) -> None:
        n = self.max_intra_counter + 1
        assert len(self.command_list) == n, f"{self.task_id}: command_list length != {n}"
        assert len(self.progress_list) == n, f"{self.task_id}: progress_list length != {n}"
        assert len(self.world_state_list) == n, f"{self.task_id}: world_state_list length != {n}"

TASK_TEMPLATE = {
    # Task 1-1 (Press the blue button N times)
    "press_button_1": TaskSpec(
        task_id="press_button_1",
        task_text="press the button one time",
        max_intra_counter=1,
        command_list=["press_button", "done"],
        progress_list=["0/1 Pressed", "1/1 Pressed"],
        world_state_list=[None, None],
    ),
    "press_button_2": TaskSpec(
        task_id="press_button_2",
        task_text="press the button two times",
        max_intra_counter=2,
        command_list=["press_button", "press_button",  "done"],
        progress_list=["0/2 Pressed", "1/2 Pressed", "2/2 Pressed"],
        world_state_list=[None, None, None],
    ),
    "press_button_3": TaskSpec(
        task_id="press_button_3",
        task_text="press the button three times",
        max_intra_counter=3,
        command_list=["press_button", "press_button", "press_button", "done"],
        progress_list=["0/3 Pressed", "1/3 Pressed", "2/3 Pressed", "3/3 Pressed"],
        world_state_list=[None, None, None, None],
    ),
#     "press_button_4": TaskSpec(
#         task_id="press_button_3",
#         task_text="press the button four times",
#         max_intra_counter=4,
#         command_list=["press_button", "press_button", "press_button", "done"],
#         progress_list=["0/3 Pressed", "1/3 Pressed", "2/3 Pressed", "3/3 Pressed"],
#         world_state_list=[None, None, None, None],
#     ),
# "press_button_5": TaskSpec(
#         task_id="press_button_5",
#         task_text="press the button three times",
#         max_intra_counter=5,
#         command_list=["press_button", "press_button", "press_button", "done"],
#         progress_list=["0/3 Pressed", "1/3 Pressed", "2/3 Pressed", "3/3 Pressed"],
#         world_state_list=[None, None, None, None],
#     ),
#     "press_rgb_in_order": TaskSpec(
#         task_id="press_rgb_in_order",
#         task_text="press the red, green, blue button in order",
#         max_intra_counter=3,
#         command_list=["press_red_button", "press_green_button", "press_blue_button", "done"],
#         progress_list=["0/3 Pressed", "1/3 Pressed", "2/3 Pressed", "3/3 Pressed"],
#         world_state_list=[None, None, None, None],
#     ),
}