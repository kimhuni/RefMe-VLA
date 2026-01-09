from dataclasses import dataclass, field
from typing import Optional, Union
from pathlib import Path

from common.policies.adalora import AdaLoraConfig


@dataclass
class ExtendedConfig:
    core: str = "vanilla"
    target_keywords: list[str] = field(default_factory=lambda: ["all-linear"])
    pretrained_expert: bool = False


    adapter_file_path: Optional[list[str | Path]] = None
    aux_loss_cfg: Optional[dict] = None
    is_train: bool = True

    expert_source: Optional[str] = "lora"

    def match_cfg(self):
        if self.core in ["vanilla"]:
            return None
        else:
            raise ValueError(f"Unknown core: {self.core}")

    @property
    def use_moe(self) -> bool:
        if self.core in ["vanilla", "lora", "qlora", "lora_ada", "qlora_ada"]:
            return False
        elif self.core in ["lora_moe", "qlora_moe", 'lora_msp', 'qlora_msp']:
            return True
        else:
            raise ValueError(f"Unknown core: {self.core}")

    @property
    def use_adapters(self) -> bool:
        return self.core in ["lora", "qlora", "lora_moe", "qlora_moe", "lora_msp", "qlora_msp", "adalora", "qadalora", "lora_ada", "qlora_ada"]