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

#  KL 散度（惩罚违规扣分项）
def diag_gaussian_kl(mu_q: torch.Tensor, sigma_q: torch.Tensor,
                    mu_p: torch.Tensor, sigma_p: torch.Tensor) -> torch.Tensor:
    """KL( N(mu_q, diag(sigma_q^2)) || N(mu_p, diag(sigma_p^2)) )
    Returns: [N] (sum over K for each row)
    算的是分布 Q (后验，看了答案的学霸) 和 分布 P (先验，没看答案的盲猜) 之间的“差距”。
    KL 散度越小，说明盲猜得越准。
    """
    # avoid division by zero
    eps = 1e-8
    # eps = 1e-8，如果 sigma 小于这个数，就强行等于这个数，防止程序因为除以 0 当场爆炸。
    sigma_q = sigma_q.clamp_min(eps)
    sigma_p = sigma_p.clamp_min(eps)

    # KL 公式的两大部分，用代码严丝合缝地翻译过来：
    # 第一部分：对数方差差值
    term1 = torch.log(sigma_p) - torch.log(sigma_q)
     # 第二部分：均值差距和方差比例
    term2 = (sigma_q.pow(2) + (mu_q - mu_p).pow(2)) / (2.0 * sigma_p.pow(2))
   
    # .sum(dim=-1)：把这 16 个因子的扣分全加起来，算出这只股票的总扣分。
    kl = (term1 + term2 - 0.5).sum(dim=-1)
    return kl


def reparameterize(mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    # 为什么要有这个？
    # 神经网络不能直接从分布中“随机抽样”，因为“随机”是不能求导（算梯度）的！
    # 所以大神们发明了这个魔术：
    # 1. torch.randn_like：先在旁边用标准正态分布瞎抽一个随机数 eps (均值0，方差1)。
    eps = torch.randn_like(mu)
    # 2. 然后用数学公式把它平移和拉伸： 
    # 这样随机性就被“隔离”到了 eps 里，模型就能顺利计算梯度去更新 mu 和 sigma 了！
    return mu + sigma * eps

# 工具 4：皮尔逊 IC（模型预测准确率的终极KPI）
def corr_ic(pred: torch.Tensor, y: torch.Tensor) -> float:
    """Pearson IC for one day; pred,y are [N]."""
    # 算分之前，先把数据从显卡(GPU)上扯下来，放到 CPU 内存里
    pred = pred.detach().cpu()
    y = y.detach().cpu()
    if pred.numel() < 3:
        return float("nan")

    # 皮尔逊相关系数的标准公式：协方差 / (各自的标准差相乘)
    vx = pred - pred.mean()
    vy = y - y.mean()
    # 分母：算波动幅度
    denom = (vx.pow(2).sum().sqrt() * vy.pow(2).sum().sqrt()).item()
    if denom == 0:
        return float("nan")# 如果今天所有股票涨幅一模一样（没波动），就没法算，返回 nan (不是一个数)
        return float("nan")
    return (vx * vy).sum().item() / denom# 分子：vx * vy 就是协方差的底子。除以分母就是最终的 IC！

# 工具 5：斯皮尔曼 Rank IC（只看排名不看数值的 KPI）
def rank_ic(pred: torch.Tensor, y: torch.Tensor) -> float:
    """Spearman RankIC for one day; pred,y are [N]."""
    pred = pred.detach().cpu()
    y = y.detach().cpu()
    
# argsort() 会返回从小到大排序的原始位置。
    # 连续调用两次 argsort()，就极其巧妙地把“数值”变成了“名次”！
    # 比如 [0.1, 0.9, 0.5] 连续两次 argsort() 后直接变成名次排名 [0, 2, 1]！
    pred_rank = pred.argsort().argsort().float()
    y_rank = y.argsort().argsort().float()
    return corr_ic(pred_rank, y_rank)

# 算年化收益率。这个咱们在前面讲第 1 个代码时，用 numpy 详细拆解过。
    # 这里只是换成了 PyTorch 的张量写法，逻辑完全一模一样！
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
    # 信息比率 (IR)：衡量你承担每一份风险，能换来多少超额收益。
    # 也是量化界发年终奖的重要考核指标！
    daily_rets = daily_rets.detach().cpu()
    if daily_rets.numel() < 2:
        return float("nan")
    mu = daily_rets.mean().item()
    sd = daily_rets.std(unbiased=True).item()
    if sd == 0:
        return float("nan")
    return (mu / sd) * math.sqrt(ann)
