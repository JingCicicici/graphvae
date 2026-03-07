from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

# 复用你项目里已有的 TopK-Drop 实现（会输出 daily_ret）
from .backtest_topkdrop import topk_drop_backtest, annualized_return, information_ratio


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.size < 3:
        return float("nan")
    sa = a.std()
    sb = b.std()
    if sa == 0 or sb == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _rank_ic(a: np.ndarray, b: np.ndarray) -> float:
    # Spearman = Pearson(rank(a), rank(b))
    ar = pd.Series(a).rank(method="average").to_numpy()
    br = pd.Series(b).rank(method="average").to_numpy()
    return _pearson(ar, br)


def _ic_stats(arr: np.ndarray, ann: int = 252) -> dict:
    arr = np.asarray([x for x in arr if np.isfinite(x)], dtype=float)
    if arr.size == 0:
        return {"mean": float("nan"), "std": float("nan"), "icir": float("nan"), "days": 0}
    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if arr.size > 1 else float("nan")
    icir = (mean / std) * math.sqrt(ann) if (arr.size > 1 and std not in (0.0, float("nan"))) else float("nan")
    return {"mean": mean, "std": std, "icir": icir, "days": int(arr.size)}


def attach_raw_labels_with_qlib(pred: pd.DataFrame, qlib_uri: str) -> pd.DataFrame:
    """
    pred columns need: date, instrument, pred, label(=z-score label from training)
    Output adds:
      - label_z : original label
      - label_raw : next-day return computed from $close
      - label : overwritten with label_raw (for backtest compatibility)
    """
    import qlib
    from qlib.data import D

    qlib.init(provider_uri=qlib_uri, region="cn")

    out = pred.copy()
    out["date"] = pd.to_datetime(out["date"])

    start = out["date"].min().date()
    end = (out["date"].max() + pd.Timedelta(days=20)).date()

    instruments = sorted(out["instrument"].dropna().unique().tolist())

    # fetch close
    close = D.features(instruments, ["$close"], start_time=str(start), end_time=str(end))
    close = close.reset_index().rename(columns={"datetime": "date"})
    close["date"] = pd.to_datetime(close["date"])
    close = close.sort_values(["instrument", "date"])

    close["label_raw"] = close.groupby("instrument")["$close"].shift(-1) / close["$close"] - 1.0

    out = out.rename(columns={"label": "label_z"})
    out = out.merge(close[["instrument", "date", "label_raw"]], on=["instrument", "date"], how="left")

    # backtest_topkdrop expects column name "label" as next-day return
    out["label"] = out["label_raw"]
    return out


def compute_excess_daily_ret(daily_abs: pd.DataFrame, qlib_uri: str, benchmark: str) -> pd.DataFrame:
    """
    daily_abs columns: date, daily_ret (already net of costs)
    Returns a df with date, daily_ret (excess = strategy - benchmark_ret)
    """
    import qlib
    from qlib.data import D

    qlib.init(provider_uri=qlib_uri, region="cn")

    d = daily_abs.copy()
    d["date"] = pd.to_datetime(d["date"])
    start = d["date"].min().date()
    end = (d["date"].max() + pd.Timedelta(days=20)).date()

    bench_close = D.features([benchmark], ["$close"], start_time=str(start), end_time=str(end))
    if len(bench_close) == 0:
        raise RuntimeError(f"Cannot fetch benchmark close for {benchmark}. Check instrument code or qlib data.")

    bench_close = bench_close.reset_index().rename(columns={"datetime": "date"})
    bench_close["date"] = pd.to_datetime(bench_close["date"])
    bench_close = bench_close.sort_values("date")
    bench_close["bench_ret"] = bench_close["$close"].shift(-1) / bench_close["$close"] - 1.0
    bench = bench_close[["date", "bench_ret"]]

    m = d.merge(bench, on="date", how="left")
    m["daily_ret"] = m["daily_ret"] - m["bench_ret"]
    return m[["date", "daily_ret"]].dropna()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_path", type=str, required=True, help="runs/.../daily_preds.parquet or csv")
    ap.add_argument("--qlib_uri", type=str, default="/root/autodl-tmp/qlib_data")
    ap.add_argument("--benchmark", type=str, default="SH000300")
    ap.add_argument("--K", type=int, default=50)
    ap.add_argument("--N", type=int, default=5)
    ap.add_argument("--buy_cost", type=float, default=0.0005)
    ap.add_argument("--sell_cost", type=float, default=0.0015)
    ap.add_argument("--out_dir", type=str, default="", help="default: same dir as pred_path")
    args = ap.parse_args()

    pred_path = Path(args.pred_path)
    out_dir = Path(args.out_dir) if args.out_dir else pred_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) load preds
    if pred_path.suffix.lower() == ".csv":
        pred = pd.read_csv(pred_path)
    else:
        pred = pd.read_parquet(pred_path)

    need = {"date", "instrument", "pred", "label"}
    miss = need - set(pred.columns)
    if miss:
        raise ValueError(f"pred file missing columns: {miss}")

    # 2) attach raw labels
    merged = attach_raw_labels_with_qlib(pred, args.qlib_uri)

    s = merged["label"].dropna()
    print("=== RAW label sanity check ===")
    print(s.describe(percentiles=[0.01, 0.05, 0.5, 0.95, 0.99]))
    print("share(|raw|>0.11):", float((s.abs() > 0.11).mean()))
    print("rows:", len(merged), "raw_non_null:", int(merged["label"].notna().sum()))

    preds_raw_path = out_dir / "daily_preds_raw.parquet"
    merged.to_parquet(preds_raw_path, index=False)
    print("Saved:", str(preds_raw_path))

    # 3) IC / RankIC on RAW return
    ics = []
    rics = []
    merged["date"] = pd.to_datetime(merged["date"])
    for dt, g in merged.groupby("date", sort=True):
        gg = g.dropna(subset=["pred", "label"])
        if len(gg) < 10:
            continue
        a = gg["pred"].to_numpy()
        b = gg["label"].to_numpy()
        ics.append(_pearson(a, b))
        rics.append(_rank_ic(a, b))

    ic_stat = _ic_stats(np.array(ics))
    ric_stat = _ic_stats(np.array(rics))

    print("\n=== IC / RankIC on RAW return ===")
    print(f"IC_mean={ic_stat['mean']:.6f} IC_std={ic_stat['std']:.6f} ICIR={ic_stat['icir']:.3f} days={ic_stat['days']}")
    print(f"RankIC_mean={ric_stat['mean']:.6f} RankIC_std={ric_stat['std']:.6f} RankICIR={ric_stat['icir']:.3f} days={ric_stat['days']}")

    # 4) TopK-Drop backtest (ABS)
    bt_df = merged[["date", "instrument", "pred", "label"]].copy()
    daily_abs = topk_drop_backtest(bt_df, K=args.K, N=args.N, buy_cost=args.buy_cost, sell_cost=args.sell_cost)
    abs_path = out_dir / "daily_ret_topkdrop_raw.parquet"
    daily_abs.to_parquet(abs_path, index=False)
    ar_abs = annualized_return(daily_abs["daily_ret"].to_numpy())
    ir_abs = information_ratio(daily_abs["daily_ret"].to_numpy())

    # 5) Excess vs benchmark
    daily_ex = compute_excess_daily_ret(daily_abs, args.qlib_uri, args.benchmark)
    ex_path = out_dir / "daily_ret_topkdrop_excess.parquet"
    daily_ex.to_parquet(ex_path, index=False)
    ar_ex = annualized_return(daily_ex["daily_ret"].to_numpy())
    ir_ex = information_ratio(daily_ex["daily_ret"].to_numpy())

    print("\n=== TopK-Drop Backtest ===")
    print(f"ABS  : AR={ar_abs:.4f} IR={ir_abs:.3f} days={len(daily_abs)}  saved={abs_path}")
    print(f"EXCESS({args.benchmark}): AR={ar_ex:.4f} IR={ir_ex:.3f} days={len(daily_ex)}  saved={ex_path}")


if __name__ == "__main__":
    main()
