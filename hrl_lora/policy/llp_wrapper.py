from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch

# Project constants (fallbacks for standalone)
try:
    from common.constants import ACTION, OBS_ROBOT
except Exception:
    ACTION = "action"
    OBS_ROBOT = "observation.state"


def _squeeze_singleton_time_dim(img: torch.Tensor) -> torch.Tensor:
    """Handle cases where video decode returns (B,1,C,H,W)."""
    if img.ndim == 5 and img.shape[1] == 1:
        return img[:, 0]
    return img


class LLPWrapper:
    """Thin wrapper around PI0/SmolVLA policy.

    Responsibilities:
      - Extract pooled image embedding for Router obs
      - Compute loss (flow matching) and perform one supervised update

    Assumes `policy.forward(batch)` returns (loss, loss_dict) like PreTrainedPolicy.
    """

    def __init__(
        self,
        policy,
        optimizer: torch.optim.Optimizer,
        *,
        scaler: Optional[torch.cuda.amp.GradScaler] = None,
        amp: bool = True,
        max_grad_norm: float = 1.0,
        device: str = "cuda",
    ):
        self.policy = policy
        self.optimizer = optimizer
        self.scaler = scaler
        self.amp = amp
        self.max_grad_norm = max_grad_norm
        self.device = device

    @torch.no_grad()
    def extract_pooled_embed_first_frame(self, batch: Dict[str, Any], image_keys: list[str]) -> torch.Tensor:
        """Return (B, D) pooled image embedding from the first frame.

        - Uses policy.prepare_images + policy.model.paligemma_with_expert.embed_image.
        - mean-pools token sequence.
        """
        self.policy.eval()

        # Build a minimal batch for prepare_images
        mini = {}
        for k in image_keys:
            if k in batch:
                img = batch[k]
                if isinstance(img, torch.Tensor):
                    img = _squeeze_singleton_time_dim(img)
                mini[k] = img

        # prepare_images expects tensors on the right device
        for k, v in mini.items():
            if isinstance(v, torch.Tensor):
                mini[k] = v.to(self.device)

        images, img_masks = self.policy.prepare_images(mini)

        # Embed each present camera and concat tokens, then pool.
        token_seqs = []
        for img, mask in zip(images, img_masks, strict=False):
            # img: (B, C, H, W)
            img_tokens = self.policy.model.paligemma_with_expert.embed_image(img)
            # img_tokens: (B, T, D)
            token_seqs.append(img_tokens)

        tok = torch.cat(token_seqs, dim=1)  # (B, sumT, D)
        pooled = tok.mean(dim=1)  # (B, D)
        return pooled

    def forward_loss(self, batch: Dict[str, Any]) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Compute loss with gradients enabled."""
        self.policy.train()
        # Move tensors to device
        batch = {k: (v.to(self.device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
        # Squeeze image singleton time dim if needed
        for k, v in list(batch.items()):
            if isinstance(v, torch.Tensor) and v.ndim == 5 and v.shape[1] == 1:
                batch[k] = v[:, 0]

        use_amp = self.amp and (self.device != "cpu")
        if use_amp:
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                loss, info = self.policy.forward(batch)
        else:
            loss, info = self.policy.forward(batch)

        return loss, info

    @torch.no_grad()
    def compute_loss_no_grad(self, batch: Dict[str, Any]) -> float:
        self.policy.eval()
        batch = {k: (v.to(self.device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
        for k, v in list(batch.items()):
            if isinstance(v, torch.Tensor) and v.ndim == 5 and v.shape[1] == 1:
                batch[k] = v[:, 0]

        use_amp = self.amp and (self.device != "cpu")
        if use_amp:
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                loss, _ = self.policy.forward(batch)
        else:
            loss, _ = self.policy.forward(batch)
        return float(loss.detach().cpu().item())

    def supervised_update_one_step(self, batch: Dict[str, Any], *, precomputed_loss: Optional[torch.Tensor] = None) -> Dict[str, float]:
        """One supervised update step for LLP (selected expert + global adapters).

        Returns logging dict.
        """
        self.optimizer.zero_grad(set_to_none=True)

        if precomputed_loss is None:
            loss, info = self.forward_loss(batch)
        else:
            loss = precomputed_loss
            info = {}

        if self.scaler is not None and self.amp and self.device != "cpu":
            self.scaler.scale(loss).backward()
            if self.max_grad_norm is not None and self.max_grad_norm > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            if self.max_grad_norm is not None and self.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()

        out = {"llp/loss": float(loss.detach().cpu().item())}
        if isinstance(info, dict):
            # add common loss fields if present
            if "l2_loss" in info:
                out["llp/l2_loss"] = float(info["l2_loss"])
        return out
