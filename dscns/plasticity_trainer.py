"""Plasticity trainer (Phase 5-C, report section 11).

Adaptive plasticity learning: P_phi itself improves from experience,

    phi_t -> phi_{t+1},

by offline, reward-weighted imitation of *successful* deltas.  Because delta
generation runs under ``no_grad`` during the loop, there is no end-to-end
gradient path into P_phi; instead, the texts of successful modifications are
stored together with the applied (successful) delta and the reward.  At
training time the case texts are **re-embedded with the current weights**,
so P_phi must reproduce a successful delta from a *drifted* internal state —
i.e. it learns a plasticity rule that generalizes, rather than memorizing
its own past outputs (report section 11.2).
"""
from __future__ import annotations

import random
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F


class PlasticityTrainer:
    """Train a network's IntrinsicPlasticityModule from successful cases."""

    def __init__(self, network: Any, config: Any, base: Any = None):
        self.network = network
        self.config = config
        self.base = base                      # provides .tokenizer for re-embedding
        self.success_memory: List[Dict[str, Any]] = []
        self.train_loss_curve: List[float] = []
        self.train_unweighted_curve: List[float] = []
        self.train_calls: int = 0
        self.optimizer = None

    def _optimizer(self):
        if self.optimizer is None:
            self.optimizer = torch.optim.Adam(
                self.network.plasticity.parameters(),
                lr=getattr(self.config, "plasticity_lr", 1e-5),
            )
        return self.optimizer

    def record_success_case(self, texts: List[str],
                            delta_params: Dict[str, torch.Tensor],
                            reward: float) -> None:
        """Store one successful modification case (reward > 0)."""
        if reward <= 0:
            return
        self.success_memory.append({
            "texts": [str(t) for t in texts],
            "delta_W_A": delta_params["delta_W_A"].detach().cpu(),
            "delta_W_B": delta_params["delta_W_B"].detach().cpu(),
            "reward": float(reward),
        })
        if len(self.success_memory) > getattr(self.config, "max_memory_size", 100):
            self.success_memory.sort(key=lambda x: x["reward"], reverse=True)
            self.success_memory = self.success_memory[:self.config.max_memory_size]

    def train_from_memory(self) -> Optional[float]:
        """Reward-weighted MSE: make P_phi reproduce successful deltas.

        Re-embeds each case's texts with the *current* adapter weights, so
        the input state has drifted since the modification was recorded; the
        module must therefore generalize the successful delta to nearby
        states instead of copying its own past output.
        """
        cfg = self.config
        if len(self.success_memory) < getattr(cfg, "min_memory_size", 10):
            return None
        if self.base is None:
            raise RuntimeError("PlasticityTrainer needs a base model (tokenizer)")
        self.network.plasticity.train()
        opt = self._optimizer()
        total_loss = 0.0
        total_unweighted = 0.0
        n_cases = 0
        batches = getattr(cfg, "plasticity_train_batches", 5)
        batch_size = getattr(cfg, "plasticity_train_batch_size", 4)
        for _ in range(batches):
            batch = random.sample(self.success_memory,
                                  min(batch_size, len(self.success_memory)))
            loss_sum = 0.0
            unweighted_sum = 0.0
            for case in batch:
                predicted = self.network.generate_delta(
                    case["texts"], self.base.tokenizer,
                    max_len=getattr(cfg, "max_len", 192),
                    grad_enabled=True)
                target = {
                    "delta_W_A": case["delta_W_A"].to(predicted["delta_W_A"].device),
                    "delta_W_B": case["delta_W_B"].to(predicted["delta_W_B"].device),
                }
                loss = F.mse_loss(predicted["delta_W_A"], target["delta_W_A"]) + \
                       F.mse_loss(predicted["delta_W_B"], target["delta_W_B"])
                loss_sum += loss * case["reward"]
                unweighted_sum += loss
            loss = loss_sum / len(batch)
            loss.backward()
            opt.step()
            opt.zero_grad()
            total_loss += float(loss.item())
            total_unweighted += float(unweighted_sum.item() / len(batch))
            n_cases += 1
        self.network.plasticity.eval()
        mean_loss = total_loss / max(1, n_cases)
        mean_unweighted = total_unweighted / max(1, n_cases)
        self.train_loss_curve.append(mean_loss)
        self.train_unweighted_curve.append(mean_unweighted)
        self.train_calls += 1
        return mean_loss

    def statistics(self) -> Dict[str, Any]:
        return {
            "memory_size": len(self.success_memory),
            "train_calls": self.train_calls,
            "train_loss_curve": self.train_loss_curve,
            "train_unweighted_curve": self.train_unweighted_curve,
            "min_memory_size": getattr(self.config, "min_memory_size", 10),
            "train_threshold": getattr(self.config, "plasticity_train_threshold", 10),
        }
