from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

Filter = Dict[str, Any]  # MVP: {"tasks": "<exact string>"} only


@dataclass(frozen=True)
class TaskSpec:
    """
    TaskSpec defines an inter-episode scenario as a grid over (inter, intra).

    Conventions (fixed for this project):
      - inter ranges 0..max_inter (inclusive)
      - max_intra[inter] is the *maximum index* (inclusive). So #states = max_intra[inter] + 1.
      - command_grid[inter][intra], progress_grid[inter][intra] are defined for all states.
      - world_state_grid[inter] is a *string* (or None). (No dict usage.)
      - task_text[inter] is the model input instruction for that inter stage.
      - episode_filters[inter] selects data episodes (recorded robot episodes) to use for that inter stage.
    """

    task_id: str
    task_text: List[str]                 # len = max_inter + 1
    episode_filters: List[Filter]        # len = max_inter + 1

    llp_commands: str

    max_inter: int                       # inter: 0..max_inter
    max_intra: List[int]                 # len = max_inter + 1; each is max index

    command_grid: List[List[str]]        # [inter][intra]
    progress_grid: List[List[str]]       # [inter][intra]
    world_state_grid: List[Optional[str]]  # [inter]  (string or None)

    def validate(self) -> None:
        mi = self.max_inter + 1

        assert len(self.task_text) == mi, (
            f"{self.task_id}: len(task_text) must be {mi}, got {len(self.task_text)}"
        )
        assert len(self.episode_filters) == mi, (
            f"{self.task_id}: len(episode_filters) must be {mi}, got {len(self.episode_filters)}"
        )
        assert len(self.max_intra) == mi, (
            f"{self.task_id}: len(max_intra) must be {mi}, got {len(self.max_intra)}"
        )
        assert len(self.command_grid) == mi, (
            f"{self.task_id}: len(command_grid) must be {mi}, got {len(self.command_grid)}"
        )
        assert len(self.progress_grid) == mi, (
            f"{self.task_id}: len(progress_grid) must be {mi}, got {len(self.progress_grid)}"
        )
        assert len(self.world_state_grid) == mi, (
            f"{self.task_id}: len(world_state_grid) must be {mi}, got {len(self.world_state_grid)}"
        )

        for inter in range(mi):
            n_states = self.max_intra[inter] + 1
            assert n_states >= 1, f"{self.task_id}: max_intra[{inter}] must be >= 0"

            cg = self.command_grid[inter]
            pg = self.progress_grid[inter]
            assert len(cg) == n_states, (
                f"{self.task_id}: command_grid[{inter}] must have {n_states} states, got {len(cg)}"
            )
            assert len(pg) == n_states, (
                f"{self.task_id}: progress_grid[{inter}] must have {n_states} states, got {len(pg)}"
            )

            f = self.episode_filters[inter]
            if f:
                assert isinstance(f, dict), f"{self.task_id}: episode_filters[{inter}] must be a dict"
                if "tasks" in f:
                    assert isinstance(f["tasks"], str), f"{self.task_id}: episode_filters[{inter}]['tasks'] must be str"

    def n_intra_states(self, inter: int) -> int:
        return self.max_intra[inter] + 1

    def get_task_text(self, inter: int) -> str:
        return self.task_text[inter]

    def get_available_llp_commands(self) -> str:
        return self.llp_commands

    def get_command(self, inter: int, intra: int) -> str:
        return self.command_grid[inter][intra]

    def get_progress(self, inter: int, intra: int) -> str:
        return self.progress_grid[inter][intra]

    def get_world_state(self, inter: int) -> Optional[str]:
        ws = self.world_state_grid[inter]
        #if ws is None:
        #    return None
        #if isinstance(ws, str) and ws.strip().lower() == "none":
        #    return None
        return str(ws)
