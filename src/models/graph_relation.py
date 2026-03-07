from __future__ import annotations
import torch
import torch.nn as nn


class GraphRelationUpdate(nn.Module):
    """Eq (8)-(9): compute relation strength eta_{i,j} then update node repr.

    Paper's attention coefficient depends only on neighbor node j:
      eta_{i,j} = softmax_{k in N_i} exp( u_a^T phi(W_a e_k + b_a) )

    We implement neighbor list as a dense [N,k] LongTensor for simplicity.
    """

    def __init__(self, hidden_dim: int, act: str = "leakyrelu"):
        super().__init__()
        self.Wa = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.ua = nn.Parameter(torch.zeros(hidden_dim))
        if act == "tanh":
            self.act = nn.Tanh()
        elif act == "leakyrelu":
            self.act = nn.LeakyReLU(0.1)
        else:
            raise ValueError(f"Unsupported act={act}")

        nn.init.normal_(self.ua, std=0.02)

    def forward(self, e: torch.Tensor, neighbors: torch.Tensor) -> torch.Tensor:
        """e: [N,H], neighbors: [N,k] -> e_hat: [N,H]"""
        N, H = e.shape
        k = neighbors.shape[1]
        if neighbors.numel() == 0:
            return e
        # node score: [N]
        h = self.act(self.Wa(e))  # [N,H]
        node_score = (h * self.ua).sum(dim=-1)  # [N]

        # gather neighbors' embeddings and scores: [N,k,H], [N,k]
        e_nb = e[neighbors]  # [N,k,H]
        s_nb = node_score[neighbors]  # [N,k]

        w = torch.softmax(s_nb, dim=1)  # [N,k]
        agg = (w.unsqueeze(-1) * e_nb).sum(dim=1)  # [N,H]
        return e + agg
