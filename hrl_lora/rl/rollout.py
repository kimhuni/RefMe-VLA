from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch

from hrl_lora.data.chunk_types import ChunkBatch


@dataclass
class StepRecord:
    obs: torch.Tensor
    act: torch.Tensor
    logp: torch.Tensor
    val: torch.Tensor
    reward: torch.Tensor
    done: torch.Tensor


class RolloutBuffer:
    def __init__(self):
        self.steps: List[StepRecord] = []

    def add(self, *, obs: torch.Tensor, act: torch.Tensor, logp: torch.Tensor, val: torch.Tensor, reward: torch.Tensor, done: torch.Tensor) -> None:
        # ensure 1D tensors
        self.steps.append(StepRecord(obs=obs.detach().cpu(), act=act.detach().cpu(), logp=logp.detach().cpu(), val=val.detach().cpu(), reward=reward.detach().cpu(), done=done.detach().cpu()))

    def __len__(self):
        return len(self.steps)

    def finalize(self, *, gamma: float, gae_lambda: float) -> Dict[str, torch.Tensor]:
        """Compute GAE advantages/returns and pack into tensors."""
        if len(self.steps) == 0:
            raise RuntimeError("RolloutBuffer is empty")

        obs = torch.cat([s.obs for s in self.steps], dim=0)
        act = torch.cat([s.act for s in self.steps], dim=0)
        logp_old = torch.cat([s.logp for s in self.steps], dim=0)
        val_old = torch.cat([s.val for s in self.steps], dim=0)
        rew = torch.cat([s.reward for s in self.steps], dim=0)
        done = torch.cat([s.done for s in self.steps], dim=0).to(dtype=torch.bool)

        n = rew.shape[0]
        adv = torch.zeros_like(rew)
        lastgaelam = 0.0

        for t in reversed(range(n)):
            if t == n - 1:
                next_nonterminal = 1.0 - float(done[t].item())
                next_value = 0.0
            else:
                next_nonterminal = 1.0 - float(done[t].item())
                next_value = val_old[t + 1].item()

            delta = rew[t].item() + gamma * next_value * next_nonterminal - val_old[t].item()
            lastgaelam = delta + gamma * gae_lambda * next_nonterminal * lastgaelam
            adv[t] = lastgaelam

        ret = adv + val_old
        return {
            "obs": obs,
            "act": act,
            "logp_old": logp_old,
            "val_old": val_old,
            "adv": adv,
            "ret": ret,
        }


class RolloutCollector:
    def __init__(
        self,
        *,
        llp_wrapper,
        moe_controller,
        router,
        image_keys: List[str],
        lambda_cons: float,
        reward_eps: float,
        device: str = "cuda",
    ):
        self.llp = llp_wrapper
        self.moe = moe_controller
        self.router = router
        self.image_keys = image_keys
        self.lambda_cons = lambda_cons
        self.reward_eps = reward_eps
        self.device = device

    def collect(self, episodes: List[List[ChunkBatch]]) -> RolloutBuffer:
        """Collect rollouts over a batch of episodes.

        Note: This implementation processes episodes sequentially with B=1 to allow per-episode expert routing.
        """
        buf = RolloutBuffer()

        for ep_chunks in episodes:
            prev_action: Optional[int] = None

            for chunk in ep_chunks:
                # Router obs: pooled image embed + time
                pooled = self.llp.extract_pooled_embed_first_frame(chunk.batch, self.image_keys)
                pooled = pooled.to(self.device)
                time_p = chunk.time_progress.to(self.device)
                obs = torch.cat([pooled, time_p.unsqueeze(-1)], dim=-1)  # (1, D+1)

                act, logp, val = self.router.act(obs)
                expert_id = int(act.item())

                # switch penalty
                switched = (prev_action is not None) and (expert_id != prev_action)
                prev_action = expert_id

                # Activate expert + trainable params
                self.moe.set_active_expert(expert_id)
                self.moe.set_trainable_for_active_expert()

                # One forward with grad: use it both for reward and supervised update
                loss, _ = self.llp.forward_loss(chunk.batch)
                loss_det = loss.detach()

                # reward = -log(loss) - lambda_cons * switch
                safe_loss = torch.clamp(loss_det, min=self.reward_eps)
                r_fit = -torch.log(safe_loss)
                r_cons = (-self.lambda_cons) * (1.0 if switched else 0.0)
                reward = r_fit + r_cons

                # supervised update (chunk마다 1회)
                _ = self.llp.supervised_update_one_step(chunk.batch, precomputed_loss=loss)

                done = chunk.done.to(self.device).to(dtype=torch.float32)

                buf.add(
                    obs=obs,
                    act=act.to(self.device),
                    logp=logp.to(self.device),
                    val=val.to(self.device),
                    reward=reward.to(self.device).view(1),
                    done=done.view(1),
                )

                if bool(chunk.done.item()):
                    prev_action = None

        return buf
