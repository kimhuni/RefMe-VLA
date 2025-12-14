from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
from torch.distributions import Categorical


class RouterNetwork(nn.Module):
    """Actor-Critic MLP Router.

    Input: obs (B, obs_dim)
      - pooled image embedding (D)
      - time progress scalar (1)

    Output:
      - logits (B, K)
      - value (B,)
    """

    def __init__(self, obs_dim: int, num_experts: int, hidden_dim: int = 512):
        super().__init__()
        self.obs_dim = obs_dim
        self.num_experts = num_experts

        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.actor = nn.Linear(hidden_dim, num_experts)
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.backbone(obs)
        logits = self.actor(h)
        value = self.critic(h).squeeze(-1)
        return logits, value

    @torch.no_grad()
    def act(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, value = self.forward(obs)
        dist = Categorical(logits=logits)
        action = dist.sample()
        logprob = dist.log_prob(action)
        return action, logprob, value

    @torch.no_grad()
    def evaluate_actions(self, obs: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, value = self.forward(obs)
        dist = Categorical(logits=logits)
        logprob = dist.log_prob(action)
        entropy = dist.entropy()
        return logprob, entropy, value
