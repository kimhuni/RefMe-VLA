# hrl_lora/models/inject_lora_moe.py
from __future__ import annotations
from typing import List, Tuple, Optional
import torch.nn as nn

from hrl_lora.configs.moe_config import MoELoRAConfig
from hrl_lora.models.moe_lora_layers import MoELoRALinear

def _iter_named_children_with_parents(root: nn.Module):
    # yield (parent, child_name, full_name, child)
    for full_name, module in root.named_modules():
        for child_name, child in module.named_children():
            child_full = f"{full_name}.{child_name}" if full_name else child_name
            yield module, child_name, child_full, child

def inject_lora_moe_inplace(root: nn.Module, cfg: MoELoRAConfig) -> List[str]:
    """
    Replace nn.Linear modules whose name ends with any cfg.target_modules
    (and optionally contains cfg.include_name_substrings) with MoELoRALinear.
    Returns list of replaced module names.
    """
    replaced: List[str] = []
    targets = tuple(cfg.target_modules)
    include = tuple(s.lower() for s in cfg.include_name_substrings) if cfg.include_name_substrings else None

    for parent, child_name, child_full, child in list(_iter_named_children_with_parents(root)):
        if not isinstance(child, nn.Linear):
            continue

        lname = child_full.lower()

        # include filter (LLM에만 붙이기)
        if include is not None and not any(s in lname for s in include):
            continue

        # target filter (q_proj/v_proj)
        if not any(lname.endswith(t.lower()) for t in targets):
            continue

        # replace
        new_mod = MoELoRALinear(
            base=child,
            num_experts=cfg.num_experts,
            r=cfg.r,
            alpha=cfg.alpha,
            dropout=cfg.dropout,
        )
        setattr(parent, child_name, new_mod)
        replaced.append(child_full)

    if len(replaced) == 0:
        raise RuntimeError(
            f"[inject_lora_moe_inplace] Replaced 0 modules. "
            f"Check include_name_substrings={cfg.include_name_substrings} and target_modules={cfg.target_modules}."
        )
    return replaced

def iter_moe_lora_modules(root: nn.Module):
    for m in root.modules():
        if isinstance(m, MoELoRALinear):
            yield m