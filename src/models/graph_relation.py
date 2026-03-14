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
        # 1. 准备一套“西装” (线性变换层)
        # nn.Linear 就是神经网络里最基础的全连接层（做矩阵乘法 Wa * x + b）。
        # 作用是：在开会前，把原本的特征 e 转换一下，提取出更高级的表达。
        self.Wa = nn.Linear(hidden_dim, hidden_dim, bias=True)
        # 2. 准备一根“打分权杖” (可学习的参数)
        # nn.Parameter 意思是告诉 PyTorch：“这是一个需要你通过不断训练来优化的参数！”
        # 它是一个长度为 hidden_dim 的向量，专门用来给股票的发言打分。
        self.ua = nn.Parameter(torch.zeros(hidden_dim))

        # 3. 准备一个“滤音器” (激活函数)
        # 神经网络如果没有激活函数，就只是一堆无聊的线性乘法。
        # LeakyReLU 的作用是：正数直接放行，负数稍微压制一下（乘以 0.1），增加非线性。
        if act == "tanh":
            self.act = nn.Tanh()
        elif act == "leakyrelu":
            self.act = nn.LeakyReLU(0.1)
        else:
            raise ValueError(f"Unsupported act={act}")
        # 给“打分权杖”随机初始化一些微小的正态分布数字，防止一开始全是 0。
        nn.init.normal_(self.ua, std=0.02)

    def forward(self, e: torch.Tensor, neighbors: torch.Tensor) -> torch.Tensor:
        """e: [N,H], neighbors: [N,k] -> e_hat: [N,H]"""
        # 进门的数据：
        # e: [N, H] -> 此时所有股票的精华特征 (比如 300只股票，每只有 64个特征)
        # neighbors: [N, k] -> 每只股票的 5 个好哥们的编号 (比如 300只股票，每只有 5个哥们)
        N, H = e.shape
        k = neighbors.shape[1]
        # 保安检查：如果连哥们都没有（k=0），那就别开会了，直接把原来的特征 e 退回去。
        if neighbors.numel() == 0:
            return e
        # 使用激活函数
        h = self.act(self.Wa(e))  # [N,H]

        # 用“打分权杖” ua 去衡量每个股票的含金量。
        # (h * self.ua) 是对应位置相乘，.sum(dim=-1) 是把这 64 个维度的分数加总成 1 个总分！
        # node_score 的形状变成了 [N]，代表这 300 只股票每个人今天的“绝对话语权分数”。
        node_score = (h * self.ua).sum(dim=-1)  # [N]

        # neighbors 是 [N, k] 的编号。这句代码直接让 PyTorch 顺着编号去 e 里面抓人！
        # 抓出来的 e_nb 变成了 3D 矩阵：[N, k, H]。
        # 意思是：这 300 只股票，每只股票面前现在都站着它的 5 个哥们，哥们手里拿着 64 个特征。
        e_nb = e[neighbors]  # [N,k,H]
        s_nb = node_score[neighbors]  # [N,k]

        # softmax(dim=1) 会在每只股票的 5 个哥们之间进行换算，把他们的分数变成 0 到 1 之间的百分比，且 5 个人加起来刚好等于 100% (即 1.0)。
        # w 就是最终的权重（谁分高，w 就越接近 1）
        w = torch.softmax(s_nb, dim=1)  # [N,k]

        # w.unsqueeze(-1) 是在给权重增加一个维度，变成 [N, k, 1]，为了能和 [N, k, H] 的特征相乘。
        # 动作：把每个哥们的特征，乘以他们分配到的百分比权重 w。
        # .sum(dim=1)：把这 5 个哥们打完折的特征加在一起，融合完毕！
        # agg (aggregation 聚合) 的形状变回了 [N, H]。这就是从朋友那里听取到的综合建议！
        agg = (w.unsqueeze(-1) * e_nb).sum(dim=1)  # [N,H]
        # 把自己原来的特征 e，加上朋友的建议 agg，得到超级特征 e_hat，返回！
        return e + agg
