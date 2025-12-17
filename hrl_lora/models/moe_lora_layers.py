# hrl_lora/models/moe_lora_layers.py
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

class ExpertLoRA(nn.Module):
    def __init__(self, in_features: int, out_features: int, r: int, alpha: int, dropout: float):
        super().__init__()
        self.r = r
        self.scaling = alpha / float(r)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.A = nn.Linear(in_features, r, bias=False)
        self.B = nn.Linear(r, out_features, bias=False)

        # init (LoRA 논문 관례: A는 작은 값, B는 0)
        nn.init.kaiming_uniform_(self.A.weight, a=5**0.5)
        nn.init.zeros_(self.B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.B(self.A(self.dropout(x))) * self.scaling

class MoELoRALinear(nn.Module):
    """
    Drop-in replacement for nn.Linear:
    y = base(x) + lora_expert[active](x)
    """
    def __init__(self, base: nn.Linear, num_experts: int, r: int, alpha: int, dropout: float):
        super().__init__()
        assert isinstance(base, nn.Linear)
        self.base = base
        self.num_experts = num_experts
        self.active_expert = 0

        # base는 기본적으로 freeze 가정 (controller에서 한번에 freeze해도 됨)
        for p in self.base.parameters():
            p.requires_grad = False

        self.experts = nn.ModuleList([
            ExpertLoRA(base.in_features, base.out_features, r=r, alpha=alpha, dropout=dropout)
            for _ in range(num_experts)
        ])

        dev = self.base.weight.device
        self.experts.to(dev)

    def set_active_expert(self, idx: int):
        self.active_expert = int(idx)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.base(x)
        y = y + self.experts[self.active_expert](x)
        return y