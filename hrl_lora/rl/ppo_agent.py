from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn


@dataclass
class PPOStats:
    loss_total: float
    loss_policy: float
    loss_value: float
    entropy: float
    approx_kl: float
    clip_frac: float


class PPOAgent:
    def __init__(
        self,
        router: nn.Module,
        optimizer: torch.optim.Optimizer,
        *,
        clip_eps: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        ppo_epochs: int = 4,
        minibatch_size: int = 256,
        max_grad_norm: float = 1.0,
        device: str = "cuda",
    ):
        self.router = router
        self.optimizer = optimizer
        self.clip_eps = clip_eps
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.ppo_epochs = ppo_epochs
        self.minibatch_size = minibatch_size
        self.max_grad_norm = max_grad_norm
        self.device = device

    def update(self, roll: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Update router using PPO.

        roll must contain:
          obs: (N, obs_dim)
          act: (N,)
          logp_old: (N,)
          adv: (N,)
          ret: (N,)
        """
        self.router.train()

        obs = roll["obs"].to(self.device)
        act = roll["act"].to(self.device)
        logp_old = roll["logp_old"].to(self.device)
        adv = roll["adv"].to(self.device)
        ret = roll["ret"].to(self.device)

        # Standard practice: normalize advantage
        adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)

        n = obs.shape[0]
        idxs = torch.arange(n, device=self.device)

        total_loss_acc = 0.0
        pol_loss_acc = 0.0
        val_loss_acc = 0.0
        ent_acc = 0.0
        kl_acc = 0.0
        clipfrac_acc = 0.0
        num_minibatches_total = 0

        for _ in range(self.ppo_epochs):
            perm = idxs[torch.randperm(n, device=self.device)]
            for start in range(0, n, self.minibatch_size):
                mb_idx = perm[start : start + self.minibatch_size]

                mb_obs = obs[mb_idx]
                mb_act = act[mb_idx]
                mb_logp_old = logp_old[mb_idx]
                mb_adv = adv[mb_idx]
                mb_ret = ret[mb_idx]

                # Recompute current-policy quantities with gradient.
                # NOTE: Some router implementations provide evaluate_actions() but may detach outputs.
                out = self.router(mb_obs)
                if isinstance(out, tuple) and len(out) == 2:
                    logits, value = out
                elif isinstance(out, dict):
                    logits = out.get("logits")
                    value = out.get("value") if "value" in out else out.get("values")
                else:
                    raise TypeError(f"Unexpected router output type: {type(out)}")

                dist = torch.distributions.Categorical(logits=logits)
                logp = dist.log_prob(mb_act)
                entropy = dist.entropy()

                # value can be (B,) or (B,1)
                if value.dim() == 2 and value.shape[-1] == 1:
                    value = value.squeeze(-1)

                ratio = torch.exp(logp - mb_logp_old)
                unclipped = ratio * mb_adv
                clipped = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * mb_adv
                loss_policy = -torch.mean(torch.min(unclipped, clipped))

                loss_value = 0.5 * torch.mean((value - mb_ret) ** 2)

                loss = loss_policy + self.value_coef * loss_value - self.entropy_coef * torch.mean(entropy)

                with torch.no_grad():
                    approx_kl = torch.mean(mb_logp_old - logp).item()
                    clipfrac = torch.mean((torch.abs(ratio - 1.0) > self.clip_eps).float()).item()

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if self.max_grad_norm is not None and self.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.router.parameters(), self.max_grad_norm)
                self.optimizer.step()

                total_loss_acc += loss.item()
                pol_loss_acc += loss_policy.item()
                val_loss_acc += loss_value.item()
                ent_acc += entropy.mean().item()
                kl_acc += approx_kl
                clipfrac_acc += clipfrac
                num_minibatches_total += 1

        denom = max(num_minibatches_total, 1)
        stats = {
            "ppo/loss_total": total_loss_acc / denom,
            "ppo/loss_policy": pol_loss_acc / denom,
            "ppo/loss_value": val_loss_acc / denom,
            "ppo/entropy": ent_acc / denom,
            "ppo/approx_kl": kl_acc / denom,
            "ppo/clip_frac": clipfrac_acc / denom,
        }
        return stats
