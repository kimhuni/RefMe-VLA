# hrl_lora/models/moe_controller.py
from __future__ import annotations
from typing import List, Dict
import torch.nn as nn

from hrl_lora.models.inject_lora_moe import iter_moe_lora_modules
from hrl_lora.models.moe_lora_layers import MoELoRALinear

class MoEController(nn.Module):
    def __init__(self, policy: nn.Module):
        super().__init__()
        self.policy = policy
        self._moe_modules: List[MoELoRALinear] = list(iter_moe_lora_modules(policy))
        if len(self._moe_modules) == 0:
            raise RuntimeError("MoEController: no MoELoRALinear found. Did you inject LoRA?")

        self.active_expert = 0

    def set_active_expert(self, idx: int):
        self.active_expert = int(idx)
        for m in self._moe_modules:
            m.set_active_expert(self.active_expert)

    def set_trainable_for_active_expert(self):
        # freeze all first
        for p in self.policy.parameters():
            p.requires_grad = False

        # unfreeze active expert LoRA params only
        for m in self._moe_modules:
            # base는 항상 freeze
            for p in m.base.parameters():
                p.requires_grad = False
            # experts 중 active만 on
            for i, expert in enumerate(m.experts):
                req = (i == self.active_expert)
                for p in expert.parameters():
                    p.requires_grad = req

    def get_trainable_params(self):
        return [p for p in self.policy.parameters() if p.requires_grad]

    def get_adapter_param_groups(self) -> List[Dict]:
        # optimizer는 "전체 LoRA 파라미터"를 들고 있어야 함 (requires_grad로 step이 결정됨)
        params = []
        for m in self._moe_modules:
            for expert in m.experts:
                params.extend(list(expert.parameters()))
        return [{"params": params}]