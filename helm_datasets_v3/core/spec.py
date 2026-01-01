from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class TaskSpecV3:
    task_id: str
    inter: int                          # max inter index (e.g., 1 means two stages: 0 and 1)
    intra: List[int]                    # per-inter max step count (e.g., [3,2])
    task_text: List[str]                # per-inter instruction text
    episode_filters: List[List[Dict[str, Any]]]  # [inter][step] -> parquet/meta filter
    memory_grid: List[List[Dict[str, str]]]      # [inter][state] (len = intra[inter] + 1)
    llp_commands: str = ""

    def validate(self) -> None:
        if len(self.task_text) != (self.inter + 1):
            raise ValueError(f"[{self.task_id}] task_text length mismatch: "
                             f"{len(self.task_text)} vs inter+1={self.inter+1}")
        if len(self.intra) != (self.inter + 1):
            raise ValueError(f"[{self.task_id}] intra length mismatch: "
                             f"{len(self.intra)} vs inter+1={self.inter+1}")
        if len(self.episode_filters) != (self.inter + 1):
            raise ValueError(f"[{self.task_id}] episode_filters length mismatch.")
        if len(self.memory_grid) != (self.inter + 1):
            raise ValueError(f"[{self.task_id}] memory_grid length mismatch.")
        if not isinstance(self.llp_commands, str):
            raise ValueError(f"[{self.task_id}] llp_commands must be a string.")

        for i in range(self.inter + 1):
            exp_states = self.intra[i] + 1
            if len(self.memory_grid[i]) != exp_states:
                raise ValueError(f"[{self.task_id}] memory_grid[{i}] length mismatch: "
                                 f"{len(self.memory_grid[i])} vs {exp_states}")
            exp_steps = self.intra[i]
            if len(self.episode_filters[i]) != exp_steps:
                raise ValueError(f"[{self.task_id}] episode_filters[{i}] steps mismatch: "
                                 f"{len(self.episode_filters[i])} vs {exp_steps}")

    def prev_curr_for_step(self, inter_idx: int, step_idx: int) -> tuple[Dict[str, str], Dict[str, str]]:
        """step_idx: 0..intra[inter]-1, map memory_grid[state=step] -> memory_grid[state=step+1]"""
        prev_mem = self.memory_grid[inter_idx][step_idx]
        curr_mem = self.memory_grid[inter_idx][step_idx + 1]
        return prev_mem, curr_mem

    def transition_prev_curr(self) -> tuple[Dict[str, str], Dict[str, str]]:
        """(0,last)->(1,0) for inter>=1"""
        if self.inter < 1:
            raise ValueError(f"[{self.task_id}] transition requires inter>=1")
        prev_mem = self.memory_grid[0][-1]
        curr_mem = self.memory_grid[1][0]
        return prev_mem, curr_mem