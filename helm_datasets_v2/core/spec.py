# helm_datasets/core/spec.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union


@dataclass
class TaskSpec:
    task_id: str
    max_inter: int
    max_intra: List[int]

    task_text: List[str]
    episode_filters: Optional[List[Dict[str, Any]]] = None

    # grids: [inter][state_index]
    command_grid: List[List[str]] = None
    progress_grid: List[List[str]] = None

    # world_state: inter별로 하나(문자열) 권장
    world_state_grid: Union[List[str], str, None] = "None"

    # v2: LLP 인터페이스 고정
    llp_command: str = ""
    init_memory: str = ""  # "Progress: ... | World_State: ..."

    def validate(self) -> None:
        assert isinstance(self.max_intra, list) and len(self.max_intra) == self.max_inter + 1
        assert isinstance(self.task_text, list) and len(self.task_text) == self.max_inter + 1

        assert isinstance(self.command_grid, list) and len(self.command_grid) == self.max_inter + 1
        assert isinstance(self.progress_grid, list) and len(self.progress_grid) == self.max_inter + 1

        for inter in range(self.max_inter + 1):
            # progress_grid는 상태가 max_intra+1개
            need_states = self.max_intra[inter] + 1
            assert len(self.progress_grid[inter]) == need_states, \
                f"[{self.task_id}] progress_grid[{inter}] must have {need_states} states"
            assert len(self.command_grid[inter]) == need_states, \
                f"[{self.task_id}] command_grid[{inter}] must have {need_states} states"

        if not self.llp_command:
            # 최소 fallback: command_grid 첫 상태를 LLP command로 간주
            self.llp_command = self.command_grid[0][0]

        if not self.init_memory:
            # 최소 fallback: 첫 progress + world_state로 구성
            ws = self.get_world_state(0)
            p0 = self.progress_grid[0][0]
            self.init_memory = f"Progress: {p0} | World_State: {ws}"

    def get_world_state(self, inter: int) -> str:
        ws = self.world_state_grid
        if isinstance(ws, list):
            w = ws[inter]
        else:
            w = ws
        if w is None or str(w).strip() == "" or str(w).strip().lower() == "none":
            return "None"
        return str(w)

    def get_task_text(self, inter: int) -> str:
        return self.task_text[inter]

    def get_llp_command(self) -> str:
        return self.llp_command

    def get_init_memory(self) -> str:
        return self.init_memory
