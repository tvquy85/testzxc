from __future__ import annotations

import torch
import torch.nn as nn


class FlowRewardV2(nn.Module):
    def __init__(self, cond_dim: int, target_dim: int = 7, hidden_dim: int = 256):
        super().__init__()
        self.cond_dim = cond_dim
        self.target_dim = target_dim
        self.net = nn.Sequential(
            nn.Linear(1 + target_dim + cond_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, target_dim),
        )

    def forward(self, t: torch.Tensor, zt: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        if t.dim() == 1:
            t = t.unsqueeze(1)
        return self.net(torch.cat([t, zt, cond], dim=1))


def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    denom = mask.sum().clamp_min(1.0)
    return (((pred - target) ** 2) * mask).sum() / denom

