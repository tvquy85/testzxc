import torch
import torch.nn as nn
import torch.nn.functional as F

class FlowRewardLite(nn.Module):
    def __init__(self, cond_dim: int, hidden_dim: int = 512, out_dim: int = 5):
        super().__init__()
        # Inputs to MLP: t (1) + zt (5) + cond (cond_dim)
        in_dim = 1 + out_dim + cond_dim
        
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, out_dim)
        )
        
    def forward(self, t: torch.Tensor, zt: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # t is (B,), need to make it (B, 1)
        if t.dim() == 1:
            t = t.unsqueeze(1)
            
        x = torch.cat([t, zt, cond], dim=1)
        v_pred = self.net(x)
        return v_pred
