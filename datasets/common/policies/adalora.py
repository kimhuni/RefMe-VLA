from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import bitsandbytes as bnb  # optional for qadalora
except Exception:
    bnb = None


def _dtype_map(dtype: str) -> torch.dtype:
    return {
        "torch.float16": torch.float16,
        "torch.bfloat16": torch.bfloat16,
        "torch.float32": torch.float32,
        "torch.uint8": torch.uint8,
    }[dtype]


@dataclass
class AdaLoraConfig:
    layer_type: str = "adalora"

    r_max: int = 16
    r_min: int = 1
    init_r: int = 8
    target_rank: int = 8  # average target rank across layers

    alpha: int = 16
    dropout: float = 0.05
    fan_in_fan_out: bool = False
    quantize: bool = False

    s_init: float = 1.0

    # allocation schedule
    warmup_steps: int = 1000
    alloc_start_step: int = 1000
    alloc_end_step: int = 5000
    alloc_interval: int = 1000

    # importance estimation
    importance_ema_decay: float = 0.97
    importance_weight: float = 0.1
    beta1: float = 0.85
    beta2: float = 0.85

    # regularization
    orth_reg_weight: float = 0.0

    # quantization
    quant_type: str = 'fp4'
    compute_dtype_: str = 'torch.bfloat16'
    compress_statistics: bool = False
    quant_storage_: str = 'torch.uint8'

    rank_change_cap_ratio: float = 1 # 0.05 # 1 means no cap

    @property
    def compute_dtype(self) -> torch.dtype:
        return _dtype_map(self.compute_dtype_)

    @property
    def quant_storage(self) -> torch.dtype:
        return _dtype_map(self.quant_storage_)


class AdaLoraLinear(nn.Module):
    def __init__(self, base: nn.Linear, cfg: AdaLoraConfig):
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError("AdaLoraLinear expects an nn.Linear to wrap")

        self.cfg = cfg
        self._load_base(base, cfg.quantize)

        in_f, out_f = self.base.in_features, self.base.out_features

        if cfg.fan_in_fan_out:
            in_f, out_f = out_f, in_f

        self.A = nn.Parameter(self.base.weight.new_zeros((out_f, cfg.r_max)))
        self.B = nn.Parameter(self.base.weight.new_zeros((cfg.r_max, in_f)))
        # self.s is already initialized to cfg.s_init (default 1.0)
        self.E = nn.Parameter(self.base.weight.new_full((cfg.r_max,), float(getattr(cfg, 's_init', 1.0))))

        self.register_buffer('mask', torch.zeros(cfg.r_max, dtype=torch.bool))
        self.mask[: max(cfg.r_min, min(cfg.init_r, cfg.r_max))] = True

        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.B, a=math.sqrt(5))

        # Official-style init: normalize U columns and V rows; s starts at 0 (scale-only)
        with torch.no_grad():
            # s = 0 (ensures ΔW = 0 at start, LoRA-like behavior)
            self.E.zero_()
            # Normalize columns of U to unit L2
            u_norm = self.A.data.norm(dim=0, keepdim=True).clamp_min(1e-12)
            self.A.data.div_(u_norm)
            # Normalize rows of V to unit L2
            v_norm = self.B.data.norm(dim=1, keepdim=True).clamp_min(1e-12)
            self.B.data.div_(v_norm)

        # NOTE: fisher_* buffers are kept for backward-compatibility but are not used
        #       by the current MS-style importance metric.
        # self.register_buffer('fisher_U', torch.zeros_like(self.A))
        # self.register_buffer('fisher_V', torch.zeros_like(self.B))
        # self.register_buffer('fisher_s', torch.zeros_like(self.E))

        # === MS repo-style EMA(|p*grad|) + uncertainty EMA buffers ===
        self.register_buffer('ema_ipt_A', torch.zeros_like(self.A))   # exp_avg_ipt for U
        self.register_buffer('ema_ipt_B', torch.zeros_like(self.B))   # exp_avg_ipt for V
        self.register_buffer('ema_ipt_E', torch.zeros_like(self.E))   # exp_avg_ipt for s
        self.register_buffer('ema_unc_A', torch.zeros_like(self.A))   # exp_avg_unc for U
        self.register_buffer('ema_unc_B', torch.zeros_like(self.B))   # exp_avg_unc for V
        self.register_buffer('ema_unc_E', torch.zeros_like(self.E))   # exp_avg_unc for s

        self.dropout = nn.Dropout(cfg.dropout) if cfg.dropout > 0.0 else nn.Identity()
        self._merged: bool = False
        self._step_count = 0

    def extra_repr(self) -> str:
        return (
            f"in_features={self.base.in_features}, out_features={self.base.out_features}, "
            f"r_max={self.cfg.r_max}, r_eff={self.effective_rank}, alpha={self.cfg.alpha}"
        )

    @property
    def effective_rank(self) -> int:
        return int(self.mask.sum().item())

    @property
    def weight(self):
        """Alias to underlying base layer's weight parameter (read-only)."""
        return self.base.weight

    def _load_base(self, base: nn.Linear, quantize: bool):
        if quantize:
            assert bnb is not None, "bitsandbytes required for quantized AdaLoRA"
            self.base = bnb.nn.Linear4bit(
                input_features=base.in_features,
                output_features=base.out_features,
                bias=base.bias is not None,
                quant_type=self.cfg.quant_type,
                compute_dtype=self.cfg.compute_dtype,
                compress_statistics=self.cfg.compress_statistics,
                quant_storage=self.cfg.quant_storage,
            )
            self.base.load_state_dict(base.state_dict())
        else:
            self.base = base
        self.base.weight.requires_grad = False

    def compute_importance_scores(self) -> torch.Tensor:
        """
        Combine U/V/s importance into per-direction scores (len=r_max):
          score = mean_over_out(ema_ipt_U * ema_unc_U)        -> shape (r_max,)
                + mean_over_in (ema_ipt_V * ema_unc_V)        -> shape (r_max,)
                + (ema_ipt_s * ema_unc_s)                     -> shape (r_max,)
        This mirrors the MS AdaLoRA RankAllocator logic where A/B/E are combined.
        """
        # ensure buffers exist (backward might not have run yet)
        A_term = (self.ema_ipt_A * (self.ema_unc_A + 1e-12)).mean(dim=0)  # (r_max,)
        B_term = (self.ema_ipt_B * (self.ema_unc_B + 1e-12)).mean(dim=1)  # (r_max,)
        E_term = (self.ema_ipt_E * (self.ema_unc_E + 1e-12)).view(-1)     # (r_max,)
        scores = A_term + B_term + E_term
        return scores

    @torch.no_grad()
    def update_importance(self) -> None:
        """
        MS AdaLoRA metric:
          ipt_now = |p * grad|
          ema_ipt = beta1 * ema_ipt + (1 - beta1) * ipt_now
          ema_unc = beta2 * ema_unc + (1 - beta2) * |ipt_now - ema_ipt|
        Call this after backward() so that p.grad is available.
        """
        if not self.training:
            return
        b1, b2 = float(self.cfg.beta1), float(self.cfg.beta2)

        # A
        if self.A.grad is not None:
            ipt_now = (self.A * self.A.grad).abs().detach()
            self.ema_ipt_A.mul_(b1).add_((1.0 - b1) * ipt_now)
            self.ema_unc_A.mul_(b2).add_((1.0 - b2) * (ipt_now - self.ema_ipt_A).abs())

        # B
        if self.B.grad is not None:
            ipt_now = (self.B * self.B.grad).abs().detach()
            self.ema_ipt_B.mul_(b1).add_((1.0 - b1) * ipt_now)
            self.ema_unc_B.mul_(b2).add_((1.0 - b2) * (ipt_now - self.ema_ipt_B).abs())

        # E
        if self.E.grad is not None:
            ipt_now = (self.E * self.E.grad).abs().detach()
            self.ema_ipt_E.mul_(b1).add_((1.0 - b1) * ipt_now)
            self.ema_unc_E.mul_(b2).add_((1.0 - b2) * (ipt_now - self.ema_ipt_E).abs())

    def step(self) -> None:
        self._step_count += 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        x_dp = self.dropout(x)
        active_s = self.E * self.mask.float()
        proj = F.linear(x_dp, self.B)  # (..., r_max)
        proj = proj * active_s.unsqueeze(0)
        # Ensure dtype consistency
        proj = proj.to(dtype=self.A.dtype)
        out = F.linear(proj, self.A)
        denom = float(self.effective_rank)  # 0일 수 있음
        scale = self.cfg.alpha / (denom + 1e-5)
        return base_out + out * scale

    @staticmethod
    @torch.no_grad()
    def reallocate_rank(model: nn.Module, step: int, total_step: int, cap_ratio: float | None = None) -> None:
        """
        MS‑style global reallocation:
        - cubic budget schedule using:
              mul_coeff = 1 - (step - initial_warmup) / (total_step - final_warmup - initial_warmup)
              curr_total = target_total + (init_total - target_total) * (mul_coeff ** 3)
        - mask_interval gating (only reallocate on those steps)
        - per-step change cap by ratio to avoid sudden loss spikes
        - global thresholding over all singular directions; per-layer r_min enforced
        """
        # 0) collect AdaLoRA layers
        layers: list[tuple[str, AdaLoraLinear]] = []
        for name, m in model.named_modules():
            if isinstance(m, AdaLoraLinear):
                layers.append((name, m))
        if not layers:
            return

        # shared cfg (assume same schedule across layers)
        _, first = layers[0]
        cfg = first.cfg

        # schedule parameters
        initial_warmup = int(getattr(cfg, "alloc_start_step", getattr(cfg, "warmup_steps", 0)))
        alloc_end = int(getattr(cfg, "alloc_end_step", int(total_step)))
        final_warmup = max(0, int(total_step) - alloc_end)  # consistent with MS repo semantics
        mask_interval = int(getattr(cfg, "alloc_interval", 1000))

        # gating: only between start and end, and on interval steps
        if step == 0:
            return
        if not (step >= initial_warmup and step <= alloc_end):
            return
        if mask_interval > 0 and ((step - initial_warmup) % mask_interval != 0):
            return

        # totals
        init_total = sum(int(m.cfg.init_r) for _, m in layers)
        target_total = int(getattr(cfg, "target_rank", 8) * len(layers))

        # 1) compute scheduled target total (MS cubic schedule)
        #mul_coeff = 1-(step-initial_warmup)/(total_step-final_warmup-initial_warmup)
        denom = max(1, (int(total_step) - int(final_warmup) - int(initial_warmup)))
        if step <= initial_warmup:
            scheduled_total = init_total
        elif step > int(total_step) - int(final_warmup):
            scheduled_total = target_total
        else:
            mul_coeff = 1.0 - ( (step - initial_warmup) / denom )
            scheduled_total = target_total + (init_total - target_total) * (mul_coeff ** 3)
            scheduled_total = int(scheduled_total)

        # 2) current total active rank
        current_total = sum(int(m.mask.sum().item()) for _, m in layers)

        # 3) per-step cap by ratio
        #cap_ratio = float(getattr(cfg, "rank_change_cap_ratio", 0.05)) if cap_ratio is None else float(cap_ratio)
        #cap = max(1, int(abs(current_total) * cap_ratio))
        delta = scheduled_total - current_total
        if delta == 0:
            return
        if delta > 0:
            target_this_step = current_total + delta #min(delta, cap)
        else:
            target_this_step = current_total + delta #- min(-delta, cap)

        # 4) gather importance scores and compute global threshold
        per_layer_scores: dict[str, torch.Tensor] = {}
        all_scores = []
        for name, m in layers:
            scores = m.compute_importance_scores()  # (r_max,)
            per_layer_scores[name] = scores
            all_scores.append(scores.view(-1))
        if not all_scores:
            return
        all_cat = torch.cat(all_scores)

        keep = int(max(0, min(target_this_step, all_cat.numel())))
        prune = all_cat.numel() - keep
        if prune <= 0:
            threshold = -float("inf")
        else:
            threshold = torch.kthvalue(all_cat, prune).values.item()

        # 5) apply masks per layer with r_min guarantee
        for name, m in layers:
            scores = per_layer_scores[name]
            desired = (scores > threshold)
            # enforce r_min
            have = int(desired.sum().item())
            need = max(0, m.cfg.r_min - have)
            if need > 0:
                topk = scores.topk(k=need).indices
                desired[topk] = True
            m.mask.copy_(desired.to(m.mask.dtype).bool())

        # optional debug log (best-effort)
        try:
            import logging
            logging.getLogger(__name__).info(
                "[adalora.ms] step=%s current=%s scheduled=%s cap=%s next_total=%s",
                step, current_total, scheduled_total, cap, target_this_step
            )
        except Exception:
            pass


    def finalize(self) -> None:
        if self.effective_rank == 0:
            return
        active = self.mask
        self.A = nn.Parameter(self.A[:, active])
        self.B = nn.Parameter(self.B[active, :])
        self.E = nn.Parameter(self.E[active])
        self.A = nn.Parameter(torch.diag(self.E) @ self.B)
        self.B = nn.Parameter(self.A)
        del self.mask
