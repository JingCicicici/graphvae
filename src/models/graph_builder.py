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
    cfg: GraphBuildConfig,# 配置清单
) -> torch.Tensor:
    assert x_window.dim() == 3, f"x_window must be [T,N,C], got {x_window.shape}"# 保安检查：形状必须是 3 维的
    T, N, C = x_window.shape# 把 T(天数), N(股票数), C(特征数) 提取出来备用
    device = x_window.device# 看看这块肉现在在哪个设备上（CPU 还是 GPU），等下新建的变量也得放在同一个地方，不然会报错。

    # 确定到底要找几个朋友？cfg.topk 默认是 5。
    # min(5, N - 1) 是防错机制：万一今天全市场停牌得只剩 3 只股票了，你非要找 5 个朋友，程序就崩了。
    # 所以最多只能找 N-1 个（排除自己）。
    k = min(cfg.topk, N - 1)
    if k <= 0:
        return torch.empty((N, 0), dtype=torch.long, device=device)

    # Ra (Relation Accumulated): 这是一个 [N, N] 的大方阵！用来记录“累加后的总关系铁不铁”。
    # 最开始全是 0。
    Ra = torch.zeros((N, N), device=device, dtype=x_window.dtype)
    # torch.eye: 生成一个对角线全是 True，其他全是 False 的方阵。代表“自己和自己”的位置。
    eye = torch.eye(N, device=device, dtype=torch.bool)
    # off: 把 eye 取反（~ 符号），变成对角线是 False，其他是 True。
    # 因为我们找朋友，不能把“自己”当成好朋友，等下要过滤掉。
    off = ~eye  # off-diagonal mask

    for t in range(T):
        # F.normalize 是把特征“归一化”（缩放到同样的尺度）。就像把大家的高矮胖瘦统一，才好比较性格。
        xt = F.normalize(x_window[t], p=2, dim=1, eps=1e-12)  # [N,C]
        # 在 PyTorch 里，@ 代表矩阵乘法。xt @ xt.T 就是让 N 只股票互相进行点乘。
        # S (Similarity): 算出来的是一个 [N, N] 的矩阵，里面装满了所有股票在这一天的“相似度”。
        # 数字越大，说明这两只股票今天走势越像！
        S = xt @ xt.T  # [N,N]

        # 挑出那些不是自己（off），且数字正常的相似度分数
        vals = S[off & torch.isfinite(S)]
        if vals.numel() == 0:
            tau = torch.tensor(0.0, device=device, dtype=S.dtype)
        else:
            # torch.quantile: 算分位数。比如 cfg.eta_quantile 是 90。
            # 这句代码就是在成千上万个相似度分数里，划出“前 10%”的及格线 tau！
            tau = torch.quantile(vals, cfg.eta_quantile / 100.0)
            tau = torch.clamp(tau, min=0.0)  # 如果及格线算出来是负数，强行提拔到 0（不要负能量朋友）

        # 4. torch.where(条件, 满足给A, 不满足给B)
        # 如果相似度大于及格线 tau，且不是自己，就保留原分数 (S)；否则强行变成 0 (torch.zeros_like)。
        R = torch.where((S > tau) & off, S, torch.zeros_like(S))

        # T-1-t 算出这是距离今天几天前。比如今天是第19天，t=0(20天前)时指数就是 19。
        # cfg.alpha_decay 比如是 0.9。20天前的感情要乘以 0.9 的 19 次方，变得非常微弱。
        # 而昨天（t=19）的感情权重就是 0.9 的 0 次方 = 1，完全保留。
        w = cfg.alpha_decay ** (T - 1 - t)  # most recent weight = 1
        Ra = Ra + w * R

    # 怕万一“自己对自己”的分数太高，强行把对角线填成极其恐怖的负数（-1e9），保证自己绝对不会被选上。
    Ra = Ra.masked_fill(eye, -1e9)

    # torch.topk：神级函数！直接在 N 只股票里，给每只股票挑出分数最高的 k 个（largest=True）。
    # .indices 的意思是：我不要他们具体考了多少分，我只要把这 k 个哥们的“学号（索引）”拿出来！
    idx = torch.topk(Ra, k=k, dim=1, largest=True, sorted=False).indices  # [N,k]
    return idx
