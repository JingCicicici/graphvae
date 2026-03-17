from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

# 复用你项目里已有的 TopK-Drop 实现（会输出 daily_ret）
from .backtest_topkdrop import topk_drop_backtest, annualized_return, information_ratio

# 之前的相关系数是用pytorch写的，不适合在正式评测使用，所以用pandaas重写一遍
def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)#asarray 的作用就是把乱七八糟的格式，统一变成 Numpy 专属的高级计算矩阵（ndarray）
    b = np.asarray(b, dtype=float)
    if a.size < 3:
        return float("nan")# 股票太少没法算
    sa = a.std()
    sb = b.std()
    if sa == 0 or sb == 0:
        return float("nan")# 没波动没法算
    return float(np.corrcoef(a, b)[0, 1])# 直接调用 numpy 底层的相关系数公式

# 算斯皮尔曼相关系数
def _rank_ic(a: np.ndarray, b: np.ndarray) -> float:
    # pd.Series.rank() 就是把具体的涨跌幅变成第1名、第2名
    ar = pd.Series(a).rank(method="average").to_numpy()
    br = pd.Series(b).rank(method="average").to_numpy()
    return _pearson(ar, br)# 把名次丢给皮尔逊算，出来的就是斯皮尔曼！

# 这是一个统计小工具：把你这几年每一天的 IC 分数汇总起来，
    # 算算你这几年平均每天 IC 是多少 (mean)，稳不稳定 (std)，以及算出 ICIR
def _ic_stats(arr: np.ndarray, ann: int = 252) -> dict:
    arr = np.asarray([x for x in arr if np.isfinite(x)], dtype=float)
    if arr.size == 0:
        return {"mean": float("nan"), "std": float("nan"), "icir": float("nan"), "days": 0}
    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if arr.size > 1 else float("nan")
    icir = (mean / std) * math.sqrt(ann) if (arr.size > 1 and std not in (0.0, float("nan"))) else float("nan")# ICIR 公式：(平均IC / IC波动) * 根号下每年交易天数
    return {"mean": mean, "std": std, "icir": icir, "days": int(arr.size)}

"""把预测表里的假标签，替换成 Qlib 数据库里的真涨跌幅"""
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

    qlib.init(provider_uri=qlib_uri, region="cn")# 1. 连接底层的 Qlib 数据库

    out = pred.copy()# 拷贝一份预测成绩单
    out["date"] = pd.to_datetime(out["date"])

    start = out["date"].min().date()
    end = (out["date"].max() + pd.Timedelta(days=20)).date()

    # 找出成绩单里到底涉及了哪些股票
    instruments = sorted(out["instrument"].dropna().unique().tolist())

    # 2. 从 Qlib 数据库里，把这些股票在这些日期的“真实收盘价($close)”全拉出来，不是拉的一堆数字，而是带有股票代码和日期编号的表格
    close = D.features(instruments, ["$close"], start_time=str(start), end_time=str(end))
    # Qlib 吐出来的数据索引很乱，用 reset_index 把它彻底拍扁成普通的表格，并把时间列改名叫 "date"
    close = close.reset_index().rename(columns={"datetime": "date"})
    close["date"] = pd.to_datetime(close["date"])# 保安查户口：确保 date 这一列是真正的时间格式，别混进字符串
    close = close.sort_values(["instrument", "date"])# 排序！先按股票代码 (instrument) 排，再按时间 (date) 从老到新排。

    # 算出这只股票明天的真实涨跌幅！
    # groupby("instrument")：按股票分组，.shift(-1)：时光机！ 把它整列数据往上提一行 。也就是说，站在“今天”这一行，你能直接拿到“明天”的收盘价！
    close["label_raw"] = close.groupby("instrument")["$close"].shift(-1) / close["$close"] - 1.0

    # 1. 之前表里那个训练用的标准化假分数，原本叫 "label"，现在给它改名叫 "label_z"
    out = out.rename(columns={"label": "label_z"})
    # 2. 用 merge，把咱们刚算好的真实收益率 "label_raw"，按日期和股票代码无缝拼接到成绩单表上。
    out = out.merge(close[["instrument", "date", "label_raw"]], on=["instrument", "date"], how="left")

    # backtest_topkdrop expects column name "label" as next-day return
    out["label"] = out["label_raw"]
    return out


def compute_excess_daily_ret(daily_abs: pd.DataFrame, qlib_uri: str, benchmark: str) -> pd.DataFrame:
    """
    扣除大盘涨幅，算出超额收益
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

    # 1. 从 Qlib 拉取大盘指数（benchmark，比如沪深300）的收盘价
    bench_close = D.features([benchmark], ["$close"], start_time=str(start), end_time=str(end))
    if len(bench_close) == 0:
        raise RuntimeError(f"Cannot fetch benchmark close for {benchmark}. Check instrument code or qlib data.")

    bench_close = bench_close.reset_index().rename(columns={"datetime": "date"})
    bench_close["date"] = pd.to_datetime(bench_close["date"])
    bench_close = bench_close.sort_values("date")
    # 2. 算大盘明天的真实涨跌幅 (bench_ret)
    bench_close["bench_ret"] = bench_close["$close"].shift(-1) / bench_close["$close"] - 1.0
    bench = bench_close[["date", "bench_ret"]]

    # 3. 把大盘的成绩，和你每天实盘赚的钱 (daily_ret) 拼在一张表里
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

    # 2. 呼叫第二板块的函数：去 Qlib 拿真实涨跌幅换掉假分数
    merged = attach_raw_labels_with_qlib(pred, args.qlib_uri)

    s = merged["label"].dropna()
    print("=== RAW label sanity check ===")
    # 看看有没有超过 11% 涨跌幅的数据。因为 A股涨跌停是 10% 左右，如果这个数字很大，说明数据脏了，有妖股或者拆股没复权！
    print(s.describe(percentiles=[0.01, 0.05, 0.5, 0.95, 0.99]))
    print("share(|raw|>0.11):", float((s.abs() > 0.11).mean()))
    print("rows:", len(merged), "raw_non_null:", int(merged["label"].notna().sum()))

    preds_raw_path = out_dir / "daily_preds_raw.parquet"
    # 把替换了真实收益率的表存下来备份
    merged.to_parquet(preds_raw_path, index=False)
    print("Saved:", str(preds_raw_path))

    # 3) IC / RankIC on RAW return
    ics = []
    rics = []
    merged["date"] = pd.to_datetime(merged["date"])
    for dt, g in merged.groupby("date", sort=True):
        gg = g.dropna(subset=["pred", "label"])#dropna：丢弃空值
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
    # daily_abs 是每天实打实赚到的钱（绝对收益 ABS）
    daily_abs.to_parquet(abs_path, index=False)
    # 算出绝对收益的 年化收益 (AR) 和 信息比率 (IR)
    ar_abs = annualized_return(daily_abs["daily_ret"].to_numpy())
    ir_abs = information_ratio(daily_abs["daily_ret"].to_numpy())

    # 5) Excess vs benchmark
    daily_ex = compute_excess_daily_ret(daily_abs, args.qlib_uri, args.benchmark)
    ex_path = out_dir / "daily_ret_topkdrop_excess.parquet"
    daily_ex.to_parquet(ex_path, index=False)
    # 算出超额收益的 年化收益 (AR) 和 信息比率 (IR)
    ar_ex = annualized_return(daily_ex["daily_ret"].to_numpy())
    ir_ex = information_ratio(daily_ex["daily_ret"].to_numpy())

    print("\n=== TopK-Drop Backtest ===")
    print(f"ABS  : AR={ar_abs:.4f} IR={ir_abs:.3f} days={len(daily_abs)}  saved={abs_path}")
    print(f"EXCESS({args.benchmark}): AR={ar_ex:.4f} IR={ir_ex:.3f} days={len(daily_ex)}  saved={ex_path}")


if __name__ == "__main__":
    main()
