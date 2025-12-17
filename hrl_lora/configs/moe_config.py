# hrl_lora/configs/moe_config.py
from dataclasses import dataclass
from typing import Tuple, Optional

@dataclass
class MoELoRAConfig:
    num_experts: int = 4
    r: int = 8
    alpha: int = 16
    dropout: float = 0.0
    target_modules: Tuple[str, ...] = ("q_proj", "v_proj")

    # 어떤 서브트리에서만 치환할지(안정성↑). None이면 전체 탐색.
    include_name_substrings: Optional[Tuple[str, ...]] = ("language_model", "gemma_expert")
    exclude_name_substrings = ("vision", "vision_tower", "vision_model")