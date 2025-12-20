from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PromptTemplate:
    """Lightweight prompt template builder."""
    system: str = "You are a helpful embodied agent."
    history_header: str = "History"
    instruction_header: str = "Task"
    observation_header: str = "Observation"
    output_header: str = "Output format"

    def render(self, task_text: str, history_text: str, observation_text: str) -> str:
        parts = []
        parts.append(f"[System]\n{self.system}")
        parts.append(f"[{self.instruction_header}]\n{task_text}")

        if history_text.strip():
            parts.append(f"[{self.history_header}]\n{history_text}")
        else:
            parts.append(f"[{self.history_header}]\n(none)")

        parts.append(f"[{self.observation_header}]\n{observation_text}")
        parts.append(
            f"[{self.output_header}]\n"
            "Return a JSON object with keys: command, progress."
        )
        return "\n\n".join(parts)


DEFAULT_TEMPLATE = PromptTemplate()