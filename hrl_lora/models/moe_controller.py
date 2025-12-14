from __future__ import annotations

from typing import Callable, List, Optional


class MoEController:
    """Controls which Expert-LoRA adapter is active.

    This class is intentionally lightweight and expects that adapters are already
    injected into `policy` (either via PEFT or your own LoRA implementation).

    Required capabilities (at least one path must exist):
      - policy.set_adapter(name or List[name])  (PEFT-style)
      - OR policy.set_active_adapter(name)      (custom)

    It also toggles requires_grad so that only global + active expert adapters are trainable.
    """

    def __init__(
        self,
        policy,
        *,
        global_adapter: str = "global",
        expert_adapters: Optional[List[str]] = None,
        param_name_filter: Optional[Callable[[str], bool]] = None,
    ):
        self.policy = policy
        self.global_adapter = global_adapter
        self.expert_adapters = expert_adapters or []
        self.active_expert_id: int | None = None

        # which params are considered "adapter params"?
        self.param_name_filter = param_name_filter or (lambda n: ("lora" in n.lower()) or ("adapter" in n.lower()))

    def set_active_expert(self, expert_id: int) -> None:
        if expert_id < 0 or expert_id >= len(self.expert_adapters):
            raise ValueError(f"expert_id {expert_id} out of range [0,{len(self.expert_adapters)-1}]")
        self.active_expert_id = expert_id

        expert_name = self.expert_adapters[expert_id]
        # Activate adapters
        if hasattr(self.policy, "set_adapter"):
            # Prefer using both global + expert simultaneously if supported by your adapter stack.
            try:
                self.policy.set_adapter([self.global_adapter, expert_name])
            except Exception:
                # Fallback: set only expert (and assume global is always-on in your implementation)
                self.policy.set_adapter(expert_name)
        elif hasattr(self.policy, "set_active_adapter"):
            self.policy.set_active_adapter(expert_name)
        else:
            raise AttributeError(
                "Policy has no adapter switching API. Implement `set_adapter` (PEFT) or `set_active_adapter` (custom)."
            )

    def set_trainable_for_active_expert(self) -> None:
        """Make only global + active expert adapter parameters trainable."""
        if self.active_expert_id is None:
            raise RuntimeError("active_expert_id is None. Call set_active_expert(expert_id) first.")

        active_name = self.expert_adapters[self.active_expert_id]
        global_name = self.global_adapter

        for name, p in self.policy.named_parameters():
            if not self.param_name_filter(name):
                # base weights stay frozen
                p.requires_grad = False
                continue

            # adapter params: enable if global or active expert
            if (global_name in name) or (active_name in name):
                p.requires_grad = True
            else:
                p.requires_grad = False

    def get_trainable_param_groups(self):
        """Return currently-trainable params (global + active expert)."""
        params = [p for p in self.policy.parameters() if getattr(p, "requires_grad", False)]
        return [{"params": params}]

    def get_adapter_param_groups(self):
        """Return *all* adapter params (global + all experts), regardless of requires_grad.

        This is the recommended way to build the LLP optimizer: include only adapter params in the optimizer,
        and let `set_trainable_for_active_expert()` toggle requires_grad to decide which expert learns.
        """
        params = []
        for name, p in self.policy.named_parameters():
            if self.param_name_filter(name):
                params.append(p)
        return [{"params": params}]
