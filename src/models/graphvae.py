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
    topk: int = 5  #找5个邻居

    # loss weights
    kl_weight: float = 1.0

# VAE 最喜欢输出概率分布（均值和方差），这个类就是专门造分布的。
class DistNet(nn.Module):
    """A small distribution network like Eq (13): LeakyReLU + Linear heads + Softplus sigma."""

    def __init__(self, in_dim: int, out_dim: int, hidden_dim: Optional[int] = None):
        super().__init__()
        h = hidden_dim or in_dim
        # 先经过一个线性层过渡一下
        self.fc = nn.Linear(in_dim, h)
        # 一路去猜均值 (mu)，也就是具体的数值
        self.mu = nn.Linear(h, out_dim)
        # 另一路去猜方差 (sigma)，也就是模型的心虚程度
        self.sigma = nn.Linear(h, out_dim)

    # Tuple（元组）：在 Python 里，它就像是一个**“用胶带封死的死板包裹”**。包裹一旦封好，里面的东西数量和顺序就绝对不能变。
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = F.leaky_relu(self.fc(x), negative_slope=0.1)# 先经过激活函数 LeakyReLU 增加非线性
        mu = self.mu(h)# 均值可以直接输出，因为收益率可正可负

        # 方差(sigma)代表波动率，波动率绝对不能是负数！
        # F.softplus 是一个神奇的函数，能把任何负数平滑地变成接近 0 的正数。
        # 后面加个 + 1e-6 是为了保底，万一算出 0，等下做除法会当场死机（除以0报错）。
        sigma = F.softplus(self.sigma(h)) + 1e-6
        return mu, sigma


class GraphVAE(nn.Module):
    """Core GraphVAE implementation from the paper (Section 3.2-3.4)."""

    def __init__(self, cfg: GraphVAEConfig):
        super().__init__()
        self.cfg = cfg
        # 把找朋友的配置单填好
        self.graph_cfg = GraphBuildConfig(cfg.eta_quantile, cfg.alpha_decay, cfg.topk)

        # 专门用来吃时间序列 (Time Series) 的神器。
        self.gru = nn.GRU(
            input_size=cfg.num_features,# 嘴巴大小：158
            hidden_size=cfg.hidden_dim,# 肚子大小：64
            num_layers=1,
            batch_first=True,# 告诉 GRU，进来的肉块第一个维度是批次（股票）
        )

        # (8)-(9) graph relation update
        self.graph_update = GraphRelationUpdate(cfg.hidden_dim, act="tanh")

        # (10)-(11) posterior encoder (per stock)
        self.post_proj = nn.Linear(cfg.hidden_dim, cfg.hidden_dim)
        self.post_mu = nn.Linear(cfg.hidden_dim, cfg.factor_dim)
        self.post_sigma = nn.Linear(cfg.hidden_dim, cfg.factor_dim)

        # 【预测专用】：考前盲猜车间 (Prior)
        # 用刚才写的 DistNet 小分队，把 64维的高级特征，变成 16维因子的均值和方差
        self.prior = DistNet(cfg.hidden_dim, cfg.factor_dim)

        # (14) decoder: alpha distribution net + beta linear
        # 【解码车间】：把 16 维因子还原成收益率 (Decoder)
        # alpha 是每只股票自带的独立收益（比如某些股天生爱涨），是个标量（1维）
        self.alpha = DistNet(cfg.hidden_dim, out_dim=1)  # scalar alpha per stock
        # beta 是这只股票对那 16 个核心因子的敏感度（暴露度）
        self.beta = nn.Linear(cfg.hidden_dim, cfg.factor_dim)

    def encode_features(self, x_window: torch.Tensor) -> torch.Tensor:
        """x_window: [T,N,C] -> e: [N,H] using GRU last hidden."""

        T, N, C = x_window.shape
        # permute 像魔方一样转动维度！把 [T, N, C] 变成 [N, T, C]
        # 必须这么转，因为 GRU 喜欢把个体（N只股票）放在第一维度。
        x = x_window.permute(1, 0, 2).contiguous()  # [N,T,C]
        # 把数据喂给 GRU。GRU 会吐出过程输出，和最后一天的心得体会 h_n
        _, h_n = self.gru(x)  # h_n: [1,N,H]

        # h_n 形状是 [1, N, H]。[-1] 就是把最外层那个多余的 1 剥掉。
        # e 就是 20 天走势浓缩成的一句话（64维向量）！
        e = h_n[-1]  # [N,H]
        return e

    def posterior(self, e_hat: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Eq (11) + mapping to (mu_post, sigma_post). y: [N]"""
        y = y.view(-1, 1)  # # 把答案 [N] 竖起来变成 [N, 1]，方便等下相乘

        # 核心：把股票的超级特征 e_hat 转换一下，然后乘以真实答案 y。
        # 也就是把“正确答案”强行融合进了特征里！
        e_cy = self.post_proj(e_hat) * y  # Hadamard with y (broadcast)

        # 用融合了答案的特征，去算出“作弊版”的均值和方差
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
        # 算出每只股票的独立收益 alpha (均值和方差)
        mu_a, sigma_a = self.alpha(e_hat)  # [N,1], [N,1]
        mu_a = mu_a.squeeze(-1)           # [N]
        sigma_a = sigma_a.squeeze(-1)     # [N]

        # 算出股票对 16 个因子的敏感度 beta [N, K]
        beta = self.beta(e_hat)           # [N,K]

        # 【核心金融公式：收益 = 独立收益 + (敏感度 * 因子收益)】
        # 这里的 mu_z 就是因子收益。 (beta * mu_z).sum(dim=-1) 是把 16 个因子的影响全加起来。
        mu_y = mu_a + (beta * mu_z).sum(dim=-1)  # [N]
        # 预测的方差 = alpha的方差 + (beta平方 * 因子方差)之和
        var_y = sigma_a.pow(2) + (beta.pow(2) * sigma_z.pow(2)).sum(dim=-1)  # [N]

        # 方差开根号，变回标准差 sigma
        sigma_y = (var_y + 1e-8).sqrt()
        return mu_y, sigma_y

    def forward(
        self,
        x_window: torch.Tensor,            # [T,N,C]
        # y 是明天的真实收益率（也就是 AI 要预测的标准答案）。
        # Optional 意思是“可以不传”。实盘预测时没答案就不传，训练做题时才传。
        y: Optional[torch.Tensor] = None,  # [N]
        neighbors: Optional[torch.Tensor] = None,  # [N,k]
    ) -> Dict[str, torch.Tensor]:# 告诉别人，这个函数最后会吐出一个字典（Dict），里面装的全是结果。
        """If y is provided, returns reconstruction and KL terms for training.
        Always returns prediction mean from prior for ranking.
        """
        # 它负责把过去 T 天的走势特征，浓缩成一个精华向量。
        # [N, H] 意思是：N 只股票，每只股票被提炼出了 H 个精华指标（比如 H=64）
        e = self.encode_features(x_window)  # [N,H]
        
        # 如果邻居没有被传进来就调用 build_neighbors_from_window 函数自己去算！
            # 它会挑出走势最相似的股票，返回一个 [N, k] 的名单（每只股票对应 k 个好哥们的编号）。
        if neighbors is None:
            neighbors = build_neighbors_from_window(x_window, self.graph_cfg)  # [N,k]
            
        # 把单只股票的精华特征 e，和它的朋友圈名单 neighbors 传进去。
        # 它会让每只股票去听听周围哥们儿的意见，然后把信息揉碎，融合出超级特征 e_hat。
        # 形状还是 [N, H]，但这时的特征已经包含了全局视角。
        e_hat = self.graph_update(e, neighbors)

        # self.prior 叫“先验分布网络”(Prior)。
        # 什么是先验？就是在没看到明天真实涨跌幅的情况下，仅凭历史经验猜一个因子的概率分布。
        # mu_prior: 模型猜出来的均值。
        # sigma_prior: 模型猜出来的方差（代表它心里有多不确定，方差越大越心虚）。
        mu_prior, sigma_prior = self.prior(e_hat)

        # mu_pred: 最终预测明天涨跌多少。
        # sigma_pred: 预测结果的不确定性。
        mu_pred, sigma_pred = self.decode(e_hat, mu_prior, sigma_prior)

        out = {
            "mu_pred": mu_pred,
            "sigma_pred": sigma_pred,
            "mu_prior": mu_prior,
            "sigma_prior": sigma_prior,
            "neighbors": neighbors,
        }
        # 如果给了真实答案 y，说明现在是在训练模型调参数！
        if y is not None:
            # self.posterior 叫“后验分布网络”(Posterior)。
            # 什么是后验？就是“看到了明天的真实答案 y 之后，事后诸葛亮地算出一个分布”。
            # 这是模型完美的“上帝视角”。
            mu_post, sigma_post = self.posterior(e_hat, y)
            # Keep reconstruction in analytic distribution form for consistency with Eq.(14).
            # 拿着上帝视角的分布去解码，算出重构（reconstruction）收益率。
            # 因为看过了答案 y，所以这个 mu_rec 理论上应该和真实的 y 极其接近！
            mu_rec, sigma_rec = self.decode(e_hat, mu_post, sigma_post)

            # diag_gaussian_kl 函数：算大名鼎鼎的 KL 散度。
            # 这是在干嘛？它在强迫“考前盲猜的分布(prior)”尽量去模仿“考后上帝视角的分布(post)”。
            # 这是这篇 FactorVAE 论文里最核心的约束机制！
            kl = diag_gaussian_kl(mu_post, sigma_post, mu_prior, sigma_prior)  # [N]

            # .pow(2) 就是平方。算上帝视角算出的值和真实值 y 之间的均方误差。
            mse = (mu_rec - y).pow(2)  # [N]

            out.update({
                "mu_post": mu_post,
                "sigma_post": sigma_post,
                "mu_rec": mu_rec,
                "sigma_rec": sigma_rec,
                "kl": kl.mean(),# 把所有股票的 KL 散度求平均
                "mse": mse.mean(),
                # 最核心的 loss：这就是深度学习模型要去努力降到最低的总目标！
                # MSE 代表“预测要准”，KL 代表“提取的因子特征要符合统计规律”。
                "loss": mse.mean() + self.cfg.kl_weight * kl.mean(),
            })

        return out
