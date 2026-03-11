from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .graph_builder import GraphBuildConfig, build_neighbors_from_window
from .graph_relation import GraphRelationUpdate
from ..utils import diag_gaussian_kl


@dataclass
class GraphVAEConfig:
    window_T: int = 20
    num_features: int = 158  # Alpha158
    hidden_dim: int = 64     # H
    factor_dim: int = 16     # K
    eta_quantile: float = 90.0
    alpha_decay: float = 0.9
    topk: int = 5

    # loss weights
    kl_weight: float = 1.0


class DistNet(nn.Module):
    """A small distribution network like Eq (13): LeakyReLU + Linear heads + Softplus sigma."""

    def __init__(self, in_dim: int, out_dim: int, hidden_dim: Optional[int] = None):
        super().__init__()
        h = hidden_dim or in_dim
        self.fc = nn.Linear(in_dim, h)
        self.mu = nn.Linear(h, out_dim)
        self.sigma = nn.Linear(h, out_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = F.leaky_relu(self.fc(x), negative_slope=0.1)
        mu = self.mu(h)
        sigma = F.softplus(self.sigma(h)) + 1e-6
        return mu, sigma


class GraphVAE(nn.Module):
    """Core GraphVAE implementation from the paper (Section 3.2-3.4)."""

    def __init__(self, cfg: GraphVAEConfig):
        super().__init__()
        self.cfg = cfg
        self.graph_cfg = GraphBuildConfig(cfg.eta_quantile, cfg.alpha_decay, cfg.topk)

        # (7) GRU feature extractor
        self.gru = nn.GRU(
            input_size=cfg.num_features,
            hidden_size=cfg.hidden_dim,
            num_layers=1,
            batch_first=True,
        )

        # (8)-(9) graph relation update
        self.graph_update = GraphRelationUpdate(cfg.hidden_dim, act="tanh")

        # (10)-(11) posterior encoder (per stock)
        self.post_proj = nn.Linear(cfg.hidden_dim, cfg.hidden_dim)
        self.post_mu = nn.Linear(cfg.hidden_dim, cfg.factor_dim)
        self.post_sigma = nn.Linear(cfg.hidden_dim, cfg.factor_dim)

        # (12)-(13) prior predictor
        self.prior = DistNet(cfg.hidden_dim, cfg.factor_dim)

        # (14) decoder: alpha distribution net + beta linear
        self.alpha = DistNet(cfg.hidden_dim, out_dim=1)  # scalar alpha per stock
        self.beta = nn.Linear(cfg.hidden_dim, cfg.factor_dim)

    def encode_features(self, x_window: torch.Tensor) -> torch.Tensor:
        """x_window: [T,N,C] -> e: [N,H] using GRU last hidden."""
        T, N, C = x_window.shape
        x = x_window.permute(1, 0, 2).contiguous()  # [N,T,C]
        _, h_n = self.gru(x)  # h_n: [1,N,H]
        e = h_n[-1]  # [N,H]
        return e

    def posterior(self, e_hat: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Eq (11) + mapping to (mu_post, sigma_post). y: [N]"""
        y = y.view(-1, 1)  # [N,1]
        e_cy = self.post_proj(e_hat) * y  # Hadamard with y (broadcast)
        mu = self.post_mu(e_cy)
        sigma = F.softplus(self.post_sigma(e_cy)) + 1e-6
        return mu, sigma

    def decode(
        self,
        e_hat: torch.Tensor,
        mu_z: torch.Tensor,
        sigma_z: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Eq (14): yhat ~ N(mu_y, sigma_y^2). Returns (mu_y, sigma_y)."""
        mu_a, sigma_a = self.alpha(e_hat)  # [N,1], [N,1]
        mu_a = mu_a.squeeze(-1)           # [N]
        sigma_a = sigma_a.squeeze(-1)     # [N]

        beta = self.beta(e_hat)           # [N,K]

        mu_y = mu_a + (beta * mu_z).sum(dim=-1)  # [N]
        var_y = sigma_a.pow(2) + (beta.pow(2) * sigma_z.pow(2)).sum(dim=-1)  # [N]
        sigma_y = (var_y + 1e-8).sqrt()
        return mu_y, sigma_y

    def forward(
        self,
        x_window: torch.Tensor,            # [T,N,C]
        y: Optional[torch.Tensor] = None,  # [N]
        neighbors: Optional[torch.Tensor] = None,  # [N,k]
    ) -> Dict[str, torch.Tensor]:
        """If y is provided, returns reconstruction and KL terms for training.
        Always returns prediction mean from prior for ranking.
        """
        e = self.encode_features(x_window)  # [N,H]

        if neighbors is None:
            neighbors = build_neighbors_from_window(x_window, self.graph_cfg)  # [N,k]

        e_hat = self.graph_update(e, neighbors)

        # prior factors
        mu_prior, sigma_prior = self.prior(e_hat)
        # Decoder follows the paper's analytic form with (mu_z, sigma_z), not sampled z.
        mu_pred, sigma_pred = self.decode(e_hat, mu_prior, sigma_prior)

        out = {
            "mu_pred": mu_pred,
            "sigma_pred": sigma_pred,
            "mu_prior": mu_prior,
            "sigma_prior": sigma_prior,
            "neighbors": neighbors,
        }

        if y is not None:
            mu_post, sigma_post = self.posterior(e_hat, y)
            # Keep reconstruction in analytic distribution form for consistency with Eq.(14).
            mu_rec, sigma_rec = self.decode(e_hat, mu_post, sigma_post)

            kl = diag_gaussian_kl(mu_post, sigma_post, mu_prior, sigma_prior)  # [N]
            mse = (mu_rec - y).pow(2)  # [N]

            out.update({
                "mu_post": mu_post,
                "sigma_post": sigma_post,
                "mu_rec": mu_rec,
                "sigma_rec": sigma_rec,
                "kl": kl.mean(),
                "mse": mse.mean(),
                "loss": mse.mean() + self.cfg.kl_weight * kl.mean(),
            })

        return out
