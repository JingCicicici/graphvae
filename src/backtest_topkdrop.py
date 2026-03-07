from __future__ import annotations

import argparse
import math
import pandas as pd
import numpy as np


def annualized_return(daily_rets: np.ndarray, ann: int = 252) -> float:
    if len(daily_rets) == 0:
        return float("nan")
    nav = float(np.prod(1.0 + daily_rets))
    years = len(daily_rets) / ann
    if years <= 0:
        return float("nan")
    return nav ** (1.0 / years) - 1.0


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
    K: int = 50,
    N: int = 5,
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

    d = df.dropna(subset=["date", "instrument", "pred", "label"]).copy()
    d["date"] = pd.to_datetime(d["date"])
    d = d.sort_values(["date", "pred"], ascending=[True, False])

    holdings: list[str] = []
    holdings_set: set[str] = set()

    rows = []

    for dt, g in d.groupby("date", sort=True):
        ranked = g["instrument"].tolist()
        score = dict(zip(g["instrument"], g["pred"]))
        ret = dict(zip(g["instrument"], g["label"]))

        if not holdings_set:
            # init: buy topK
            holdings = ranked[:K]
            holdings_set = set(holdings)
            n_buy, n_sell = len(holdings), 0
        else:
            # sell N lowest-score in holdings
            hold_sorted = sorted(holdings, key=lambda x: score.get(x, -1e18))
            sell = hold_sorted[: min(N, len(hold_sorted))]

            # buy N highest-score not held
            buy = []
            for x in ranked:
                if x not in holdings_set:
                    buy.append(x)
                if len(buy) >= N:
                    break

            for x in sell:
                holdings_set.remove(x)
            for x in buy:
                holdings_set.add(x)

            # refresh holdings list (keep size K as much as possible)
            holdings = [x for x in holdings if x not in set(sell)] + buy
            # if holdings accidentally not K (e.g. short universe), fix:
            if len(holdings) > K:
                holdings = holdings[:K]
                holdings_set = set(holdings)

            n_buy, n_sell = len(buy), len(sell)

        # portfolio daily return: equal-weight on holdings
        if len(holdings) == 0:
            port_ret = 0.0
        else:
            port_ret = float(np.mean([ret.get(x, 0.0) for x in holdings]))

        # transaction cost proportional to turnover fraction
        cost = (n_buy / max(K, 1)) * buy_cost + (n_sell / max(K, 1)) * sell_cost
        rows.append({"date": dt.strftime("%Y-%m-%d"), "daily_ret": port_ret - cost})

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
