from __future__ import annotations

import argparse
import json
import os

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .data.dataset import load_panel_npz, SlidingWindowDataset
from .utils import set_seed, corr_ic, rank_ic

# 核心函数 1：制造“滚动时间窗口” (应对股市的概念漂移)
def make_rolling_splits(dates, train_years=5, valid_years=1, test_years=1):
    years = sorted({d.year for d in dates})
    for end_train_year in range(years[0] + train_years - 1, years[-1] - (valid_years + test_years) + 1):
        train_start = end_train_year - train_years + 1
        valid_year = end_train_year + 1
        test_year = end_train_year + 2

        # 找出现实世界里，属于这些年份的日期的具体“行号(idx)”
        train_idx = [i for i, d in enumerate(dates) if train_start <= d.year <= end_train_year]
        valid_idx = [i for i, d in enumerate(dates) if d.year == valid_year]
        test_idx = [i for i, d in enumerate(dates) if d.year == test_year]

        if len(train_idx) == 0 or len(valid_idx) == 0 or len(test_idx) == 0:
            continue

        # yield 是生成器黑魔法：不一次性全返回，而是每次外部调用 for 循环时，吐出一组 (起点, 终点)
        yield (min(train_idx), max(train_idx)), (min(valid_idx), max(valid_idx)), (min(test_idx), max(test_idx))


class GRUBaseline(nn.Module):
    """No-graph: per-stock GRU over window [T,C] -> hidden -> linear -> pred."""
    """No-graph: 每只股票单独丢进 GRU 算时间特征，然后直接连一个线性层预测。"""
    def __init__(self, in_dim: int, hidden: int = 64, num_layers: int = 1, dropout: float = 0.0):
        super().__init__()
        self.gru = nn.GRU(
            input_size=in_dim,# 输入的因子数量 (比如 158)
            hidden_size=hidden,# 隐层神经元数量
            num_layers=num_layers,
            batch_first=True,# 极其关键：告诉 PyTorch 传进来的数据第一维是 Batch(这里是股票数量N)
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden, 1)# 定义一个全连接层（打分器），把隐藏层压缩成 1 个预测分

    def forward(self, xw: torch.Tensor) -> torch.Tensor:
        """
        【调用目的】：模型的前向传播，定义数据进出模型的路线。
        xw 的初始形状: [T(天数), N(股票数), C(因子数)]
        """
        # -> [N, T, C]
        # permute(1,0,2) 就是矩阵转置，把形状变成 [N, T, C]，这样才符合上面 batch_first=True 的要求！
        x = xw.permute(1, 0, 2).contiguous()
        # 丢进 GRU 提纯，out 是每一天的隐藏状态
        out, _ = self.gru(x)          # [N, T, H]
        # 量化交易的核心：我们只用过去 20 天里【最后一天】的记忆，去预测明天！
        h = out[:, -1, :]             # last step [N, H]
        # 用全连接层打分，并把多余的 1 维捏扁
        pred = self.head(h).squeeze(-1)# 最终吐出: [N] (300只股票各自的预测分)
        return pred


def train_one_split(model, train_loader, valid_loader, device, lr=1e-3, epochs=30):
    opt = torch.optim.Adam(model.parameters(), lr=lr)# 实例化一个 Adam 优化器
    best = float("inf") # 记录历史上最好的验证集 Loss
    best_state = None # 记录最好的一次模型“脑子”(参数)

    for ep in range(1, epochs + 1):
        model.train()
        for xw, y, _, _ in train_loader:
            xw = xw.squeeze(0).to(device)  # [T,N,C]
            y = y.squeeze(0).to(device)    # [N]
            pred = model(xw)
            loss = torch.mean((pred - y) ** 2)

            opt.zero_grad()
            loss.backward()
            opt.step()

        model.eval()
        vloss = 0.0
        n = 0
        with torch.no_grad():
            for xw, y, _, _ in valid_loader:
                xw = xw.squeeze(0).to(device)
                y = y.squeeze(0).to(device)
                pred = model(xw)
                loss = torch.mean((pred - y) ** 2)
                vloss += float(loss.item())
                n += 1
        vloss /= max(n, 1)  # 算个平均分

        if vloss < best:
            best = vloss
            # 把这一刻的参数单独复制一份存起来 (因为后面可能越学越差)
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

        print(f"epoch={ep:03d} valid_loss={vloss:.6f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best

# 核心函数 4：吐出最终成绩单 (测试集推理)
def predict_daily(model, loader, device, instruments, split_id):
    model.eval()
    rows_metrics, rows_preds = [], []

    with torch.no_grad():
        for batch_idx, (xw, y, date_str, inst_idx) in enumerate(loader):
            xw = xw.squeeze(0).to(device)
            y = y.squeeze(0).to(device)

            if isinstance(inst_idx, torch.Tensor):
                inst_idx = inst_idx.squeeze(0)

            pred = model(xw).detach().cpu().flatten()
            y_true = y.detach().cpu().flatten()

            if isinstance(date_str, (list, tuple)):
                date_str = date_str[0]

            # 呼叫 utils.py 里的工具，算一下今天的 IC 和 RankIC，存进指标表
            ic = corr_ic(pred, y_true)
            ric = rank_ic(pred, y_true)
            rows_metrics.append({"date": date_str, "batch_idx": batch_idx, "ic": ic, "rank_ic": ric, "split": split_id})

            idx_list = inst_idx.detach().cpu().tolist()# 把这 300 只股票具体的分数和对应的真实代码（比如茅台），一行行拼接成成绩单！
            for gidx, p, r in zip(idx_list, pred.tolist(), y_true.tolist()):
                rows_preds.append({
                    "date": date_str,
                    "inst_idx": int(gidx),
                    "instrument": str(instruments[int(gidx)]),
                    "pred": float(p),
                    "label": float(r),   # label_z; eval_full 会替换成 raw
                    "split": split_id,
                })

    return pd.DataFrame(rows_metrics), pd.DataFrame(rows_preds)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, default="")
    p.add_argument("--market", type=str, default="csi300")
    p.add_argument("--npz_path", type=str, default="")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda")

    p.add_argument("--window_T", type=int, default=20)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--num_layers", type=int, default=1)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--epochs", type=int, default=30)

    p.add_argument("--run_name", type=str, default="")
    args = p.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    npz_path = args.npz_path or os.path.join(args.data_dir, f"{args.market}_alpha158_2013-01-01_2023-12-31.npz")
    panel = load_panel_npz(npz_path)

    run_dir = os.path.join("runs", args.market, args.run_name) if args.run_name else os.path.join("runs", args.market)
    os.makedirs(run_dir, exist_ok=True)
    try:
        with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as f:
            json.dump({"args": vars(args), "npz_path": npz_path}, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

    C = int(panel.features.shape[-1])# 看看有多少个因子特征
    all_m, all_p = [], []

    # 3. 核心大循环！调用 make_rolling_splits，不断地向时间轴后面滚。
    for split_id, (train_rg, valid_rg, test_rg) in enumerate(make_rolling_splits(panel.dates), start=1):
        print(f"\n=== split {split_id}: train={train_rg} valid={valid_rg} test={test_rg} ===")

        # 每次滚到新的一年，都要【重新初始化一个全新脑子】的 GRU 模型！
        model = GRUBaseline(C, hidden=args.hidden, num_layers=args.num_layers, dropout=args.dropout).to(device)

        train_ds = SlidingWindowDataset(panel, args.window_T, train_rg[0], train_rg[1])
        valid_ds = SlidingWindowDataset(panel, args.window_T, valid_rg[0], valid_rg[1])
        test_ds  = SlidingWindowDataset(panel, args.window_T, test_rg[0], test_rg[1])

        train_loader = DataLoader(train_ds, batch_size=1, shuffle=True)
        valid_loader = DataLoader(valid_ds, batch_size=1, shuffle=False)
        test_loader  = DataLoader(test_ds,  batch_size=1, shuffle=False)

        model, best_v = train_one_split(model, train_loader, valid_loader, device, lr=args.lr, epochs=args.epochs)

        # 练完后，拿测试集去算最终成绩单 (m_df是每天的指标表，p_df是具体的预测明细表)
        m_df, p_df = predict_daily(model, test_loader, device, panel.instruments, split_id)
        # 把这一年的成绩单追加到总账本里，然后开始滚到下一年...
        all_m.append(m_df)
        all_p.append(p_df)

    # 4. 所有年份都滚完了，把零碎的账本拼成一张巨大无比的表
    metrics = pd.concat(all_m, ignore_index=True)
    preds   = pd.concat(all_p, ignore_index=True)

    metrics.to_parquet(os.path.join(run_dir, "daily_metrics.parquet"), index=False)
    preds.to_parquet(os.path.join(run_dir, "daily_preds.parquet"), index=False)
    preds.to_csv(os.path.join(run_dir, "daily_preds.csv"), index=False)

    print(f"Saved metrics to: {os.path.join(run_dir, 'daily_metrics.parquet')}")
    print(f"Saved preds to: {os.path.join(run_dir, 'daily_preds.parquet')}")


if __name__ == "__main__":
    main()
