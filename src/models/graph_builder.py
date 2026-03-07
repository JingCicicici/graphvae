from __future__ import annotations
from dataclasses import dataclass
import torch
import torch.nn.functional as F


@dataclass
class GraphBuildConfig:
    eta_quantile: float = 90.0
    alpha_decay: float = 0.9
    topk: int = 5


@torch.no_grad()
def build_neighbors_from_window(
    x_window: torch.Tensor,  # [T, N, C]
    cfg: GraphBuildConfig,
) -> torch.Tensor:
    assert x_window.dim() == 3, f"x_window must be [T,N,C], got {x_window.shape}"
    T, N, C = x_window.shape
    device = x_window.device

    k = min(cfg.topk, N - 1)
    if k <= 0:
        return torch.empty((N, 0), dtype=torch.long, device=device)

    Ra = torch.zeros((N, N), device=device, dtype=x_window.dtype)
    eye = torch.eye(N, device=device, dtype=torch.bool)
    off = ~eye  # off-diagonal mask

    for t in range(T):
        xt = F.normalize(x_window[t], p=2, dim=1, eps=1e-12)  # [N,C]
        S = xt @ xt.T  # [N,N]

        # quantile threshold on off-diagonal finite values
        vals = S[off & torch.isfinite(S)]
        if vals.numel() == 0:
            tau = torch.tensor(0.0, device=device, dtype=S.dtype)
        else:
            tau = torch.quantile(vals, cfg.eta_quantile / 100.0)
            tau = torch.clamp(tau, min=0.0)  # paper says positive threshold

        # keep edges only if S > tau and not self-edge
        R = torch.where((S > tau) & off, S, torch.zeros_like(S))

        w = cfg.alpha_decay ** (T - 1 - t)  # most recent weight = 1
        Ra = Ra + w * R

    # top-k neighbors per node (exclude self)
    Ra = Ra.masked_fill(eye, -1e9)
    idx = torch.topk(Ra, k=k, dim=1, largest=True, sorted=False).indices  # [N,k]
    return idx
