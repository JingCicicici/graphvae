from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple
import math
import torch


def set_seed(seed: int = 42) -> None:
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def diag_gaussian_kl(mu_q: torch.Tensor, sigma_q: torch.Tensor,
                    mu_p: torch.Tensor, sigma_p: torch.Tensor) -> torch.Tensor:
    """KL( N(mu_q, diag(sigma_q^2)) || N(mu_p, diag(sigma_p^2)) )
    Returns: [N] (sum over K for each row)
    """
    # avoid division by zero
    eps = 1e-8
    sigma_q = sigma_q.clamp_min(eps)
    sigma_p = sigma_p.clamp_min(eps)

    # KL per dimension
    term1 = torch.log(sigma_p) - torch.log(sigma_q)
    term2 = (sigma_q.pow(2) + (mu_q - mu_p).pow(2)) / (2.0 * sigma_p.pow(2))
    kl = (term1 + term2 - 0.5).sum(dim=-1)
    return kl


def reparameterize(mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    eps = torch.randn_like(mu)
    return mu + sigma * eps


def corr_ic(pred: torch.Tensor, y: torch.Tensor) -> float:
    """Pearson IC for one day; pred,y are [N]."""
    pred = pred.detach().cpu()
    y = y.detach().cpu()
    if pred.numel() < 3:
        return float("nan")
    vx = pred - pred.mean()
    vy = y - y.mean()
    denom = (vx.pow(2).sum().sqrt() * vy.pow(2).sum().sqrt()).item()
    if denom == 0:
        return float("nan")
    return (vx * vy).sum().item() / denom


def rank_ic(pred: torch.Tensor, y: torch.Tensor) -> float:
    """Spearman RankIC for one day; pred,y are [N]."""
    pred = pred.detach().cpu()
    y = y.detach().cpu()

    pred_rank = pred.argsort().argsort().float()
    y_rank = y.argsort().argsort().float()
    return corr_ic(pred_rank, y_rank)


def annualized_return(daily_rets: torch.Tensor, ann: int = 252) -> float:
    # daily_rets: [D]
    daily_rets = daily_rets.detach().cpu()
    if daily_rets.numel() == 0:
        return float("nan")
    nav = (1.0 + daily_rets).prod().item()
    years = daily_rets.numel() / ann
    if years <= 0:
        return float("nan")
    return nav ** (1.0 / years) - 1.0


def information_ratio(daily_rets: torch.Tensor, ann: int = 252) -> float:
    daily_rets = daily_rets.detach().cpu()
    if daily_rets.numel() < 2:
        return float("nan")
    mu = daily_rets.mean().item()
    sd = daily_rets.std(unbiased=True).item()
    if sd == 0:
        return float("nan")
    return (mu / sd) * math.sqrt(ann)
