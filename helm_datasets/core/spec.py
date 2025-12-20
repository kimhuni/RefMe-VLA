from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional


@dataclass(frozen=True)
class TaskSpec:
    """Defines a task as a 2D grid over (inter, intra),
    plus a persistent world_state over inter only.
    """

    task_id: str
    task_text: str

    max_inter: int  # inter ranges 0..max_inter
    max_intra: int  # intra ranges 0..max_intra (for inter=0; commit inter는 states()에서 제한 가능)

    command_grid: List[List[str]]   # [inter][intra]
    progress_grid: List[List[str]]  # [inter][intra]

    # ✅ world_state는 inter가 바뀔 때만 바뀌는 것으로 정의
    world_state_grid: List[Optional[str]]  # [inter]

    def validate(self) -> None:
        mi = self.max_inter + 1
        ni = self.max_intra + 1

        # command/progress grid: 기본은 직사각형으로 체크
        assert len(self.command_grid) == mi, f"{self.task_id}: command_grid rows != {mi}"
        assert len(self.progress_grid) == mi, f"{self.task_id}: progress_grid rows != {mi}"

        for r in range(mi):
            assert len(self.command_grid[r]) == ni, f"{self.task_id}: command_grid[{r}] cols != {ni}"
            assert len(self.progress_grid[r]) == ni, f"{self.task_id}: progress_grid[{r}] cols != {ni}"

        # ✅ world_state_grid는 inter 축만 체크
        assert len(self.world_state_grid) == mi, f"{self.task_id}: world_state_grid length != {mi}"

    def states(self) -> List[Tuple[int, int]]:
        """Enumerate states to build.
        기본 정책: inter=1(커밋)은 intra=0만 만든다.
        (원하면 여기 정책을 더 일반화할 수도 있음)
        """
        out: List[Tuple[int, int]] = []
        for inter in range(self.max_inter + 1):
            if self.max_inter >= 1 and inter == 1:
                out.append((inter, 0))
            else:
                for intra in range(self.max_intra + 1):
                    out.append((inter, intra))
        return out

    def get_command(self, inter: int, intra: int) -> str:
        return self.command_grid[inter][intra]

    def get_progress(self, inter: int, intra: int) -> str:
        return self.progress_grid[inter][intra]

    def get_world_state(self, inter: int) -> Optional[str]:
        return self.world_state_grid[inter]

    def is_commit_state(self, inter: int, intra: int) -> bool:
        """Default convention: (inter=1, intra=0) is the commit step if max_inter >= 1."""
        return self.max_inter >= 1 and inter == 1 and intra == 0


def repeat_row(row: List[str], n_rows: int) -> List[List[str]]:
    """Utility for making a grid by repeating a single row."""
    return [list(row) for _ in range(n_rows)]