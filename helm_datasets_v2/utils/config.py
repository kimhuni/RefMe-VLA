from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from helm_datasets.utils.task_template import TaskSpec, TASK_TEMPLATE

def get_default_meta(task_spec: TaskSpec) -> Dict[str, Any]:
    return {
        "task_id": task_spec.task_id,
        "task_text": task_spec.task_text,
        "fps_frames": 1,
        "cameras": ["table", "wrist"],
        "format": "sharegpt",
        "assistant_output_format": "yaml",
        "schema_version": 1,
    }

def get_task_registry() -> Dict[str, TaskSpec]:
    """
    템플릿을 JSON/YAML로 빼지 않고, python config로 관리.
    task마다 progress/command/world_state 전이표(state machine)를 명시적으로 둔다.
    """
    registry: Dict[str, TaskSpec] = TASK_TEMPLATE
    # validate
    for _, spec in registry.items():
        spec.validate()
    return registry