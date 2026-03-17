from __future__ import annotations

import argparse
import math
import pandas as pd
import numpy as np

# 赚钱指标 1：年化收益率 (Annualized Return)
def annualized_return(daily_rets: np.ndarray, ann: int = 252) -> float:
    # daily_rets 是你每天赚的钱（比如 0.01 就是赚 1%）。ann=252 是 A 股一年的交易天数。
    if len(daily_rets) == 0:
        return float("nan")
    # np.prod 是连乘！(1 + 第一天收益) * (1 + 第二天收益)... 
    # 算出来的 nav (Net Asset Value) 就是你的最终净值。比如 1.5 就是赚了 50%。
    nav = float(np.prod(1.0 + daily_rets))
    # 算出你一共炒了几年股
    years = len(daily_rets) / ann
    if years <= 0:
        return float("nan")
    return nav ** (1.0 / years) - 1.0# 资金的年化复利公式：(最终净值 ^ (1/年数)) - 1


def information_ratio(daily_rets: np.ndarray, ann: int = 252) -> float:
    if len(daily_rets) < 2:
        return float("nan")
    mu = float(np.mean(daily_rets))
    sd = float(np.std(daily_rets, ddof=1))
    if sd == 0:
        return float("nan")
    return (mu / sd) * math.sqrt(ann)


def topk_drop_backtest(
    df: pd.DataFrame,
    K: int = 50,# 我的手提箱大小：永远只拿 50 只股票
    N: int = 5,# 每天换血额度：最多只能踢掉 5 只，买入 5 只
    buy_cost: float = 0.0005,
    sell_cost: float = 0.0015,
) -> pd.DataFrame:
    """
    TopK-Drop strategy:
      - Maintain K stocks.
      - Each day: sell N stocks with lowest score among holdings; buy N highest-score stocks not held.
      - daily return = average(next-day returns of holdings) - proportional transaction cost.
    df columns: date, instrument, pred(score), label(next-day return)
    """
    need = {"date", "instrument", "pred", "label"}
    miss = need - set(df.columns)
    if miss:
        raise ValueError(f"missing columns: {miss}")

    # ... 保安查户口，确保表格里有日期、股票代码、预测分、真实收益 ...
    d = df.dropna(subset=["date", "instrument", "pred", "label"]).copy()
    d["date"] = pd.to_datetime(d["date"])
    d = d.sort_values(["date", "pred"], ascending=[True, False])

    holdings: list[str] = []
    holdings_set: set[str] = set()# 方便快速查找的集合

    rows = []# 记账本，记录每天赚了多少

    # groupby("date")：时间的车轮滚滚向前，每一天循环一次！
    for dt, g in d.groupby("date", sort=True):
        ranked = g["instrument"].tolist()
        # score 和 ret: 查分字典和查真实收益的字典
        score = dict(zip(g["instrument"], g["pred"]))
        ret = dict(zip(g["instrument"], g["label"]))

        # 如果是开盘第一天，手里没货，直接把排名前 K (50只) 的股票全买了！
        if not holdings_set:
            # init: buy topK
            holdings = ranked[:K]
            holdings_set = set(holdings)
            n_buy, n_sell = len(holdings), 0# 记录今天交易了多少笔，等下要扣手续费
        else:
            # ============ 日常换仓操作 ============
            
            # 1. 揪出差生：把手里的股票按今天的分数重新排个序，找出分数最低的 N 只股票准备卖掉
            hold_sorted = sorted(holdings, key=lambda x: score.get(x, -1e18))
            sell = hold_sorted[: min(N, len(hold_sorted))]

            # 2. 挑尖子生
            buy = []
            for x in ranked:
                if x not in holdings_set:
                    buy.append(x)
                if len(buy) >= N:
                    break

            # 3. 执行交易：把差生踢出名单，把尖子生加进名单
            for x in sell:
                holdings_set.remove(x)
            for x in buy:
                holdings_set.add(x)

            # refresh holdings list (keep size K as much as possible)
            holdings = [x for x in holdings if x not in set(sell)] + buy# 刷新我的手提箱
            # if holdings accidentally not K (e.g. short universe), fix:
            if len(holdings) > K:
                holdings = holdings[:K]
                holdings_set = set(holdings)

            n_buy, n_sell = len(buy), len(sell)

        # portfolio daily return: equal-weight on holdings
        if len(holdings) == 0:
            port_ret = 0.0
        else:
            port_ret = float(np.mean([ret.get(x, 0.0) for x in holdings]))# 计算今天我的股票池（手提箱里的 50 只股票）的平均真实涨跌幅

        # transaction cost proportional to turnover fraction
        cost = (n_buy / max(K, 1)) * buy_cost + (n_sell / max(K, 1)) * sell_cost# 扣除今天买卖折腾掉的手续费成本
        rows.append({"date": dt.strftime("%Y-%m-%d"), "daily_ret": port_ret - cost})# 把今天扣完税真实赚的钱，记在账本上！

    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_path", type=str, required=True, help="daily_preds.parquet or csv")
    ap.add_argument("--out_path", type=str, required=True, help="output parquet/csv with columns date,daily_ret")
    ap.add_argument("--K", type=int, default=50)
    ap.add_argument("--N", type=int, default=5)
    ap.add_argument("--buy_cost", type=float, default=0.0005)
    ap.add_argument("--sell_cost", type=float, default=0.0015)
    args = ap.parse_args()

    if args.pred_path.endswith(".csv"):
        df = pd.read_csv(args.pred_path)
    else:
        df = pd.read_parquet(args.pred_path)

    daily = topk_drop_backtest(df, K=args.K, N=args.N, buy_cost=args.buy_cost, sell_cost=args.sell_cost)

    # save
    if args.out_path.endswith(".csv"):
        daily.to_csv(args.out_path, index=False)
    else:
        daily.to_parquet(args.out_path, index=False)

    rets = daily["daily_ret"].to_numpy()
    ar = annualized_return(rets)
    ir = information_ratio(rets)

    print("Saved:", args.out_path)
    print("days:", len(daily))
    print("AR:", ar, "IR:", ir)
    print(daily.head())


if __name__ == "__main__":
    main()
