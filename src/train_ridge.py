from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .data.dataset import load_panel_npz, SlidingWindowDataset
from .utils import set_seed, corr_ic, rank_ic


def make_rolling_splits(dates, train_years=5, valid_years=1, test_years=1):
    years = sorted({d.year for d in dates})
    for end_train_year in range(years[0] + train_years - 1, years[-1] - (valid_years + test_years) + 1):
        train_start = end_train_year - train_years + 1
        valid_year = end_train_year + 1
        test_year = end_train_year + 2

        train_idx = [i for i, d in enumerate(dates) if train_start <= d.year <= end_train_year]
        valid_idx = [i for i, d in enumerate(dates) if d.year == valid_year]
        test_idx = [i for i, d in enumerate(dates) if d.year == test_year]

        if len(train_idx) == 0 or len(valid_idx) == 0 or len(test_idx) == 0:
            continue

        yield (min(train_idx), max(train_idx)), (min(valid_idx), max(valid_idx)), (min(test_idx), max(test_idx))


def fit_ridge_stream(train_loader: DataLoader, C: int, lam: float) -> np.ndarray:
    """
    Streamingly accumulate X'X and X'y to fit ridge:
      w = (X'X + lam I)^(-1) X'y
    Feature per stock: last day in window (xw[-1]) -> shape [N_s, C]
    """
    XtX = np.zeros((C, C), dtype=np.float64)
    Xty = np.zeros((C,), dtype=np.float64)

    for xw, y, _, _ in train_loader:
        # batch_size=1
        xw = xw.squeeze(0).numpy()   # [T, N_s, C]
        y = y.squeeze(0).numpy()     # [N_s]
        X = xw[-1].astype(np.float64)  # [N_s, C]
        yy = y.astype(np.float64)      # [N_s]

        XtX += X.T @ X
        Xty += X.T @ yy

    A = XtX + lam * np.eye(C, dtype=np.float64)
    w = np.linalg.solve(A, Xty)
    return w.astype(np.float32)


def predict_daily_ridge(w: np.ndarray, loader: DataLoader, instruments: list[str], split_id: int):
    rows_metrics = []
    rows_preds = []

    for batch_idx, (xw, y, date_str, inst_idx) in enumerate(loader):
        xw = xw.squeeze(0).numpy()   # [T, N_s, C]
        y = y.squeeze(0).numpy()     # [N_s]

        if isinstance(inst_idx, torch.Tensor):
            inst_idx = inst_idx.squeeze(0).detach().cpu().tolist()
        else:
            inst_idx = list(inst_idx)

        if isinstance(date_str, (list, tuple)):
            date_str = date_str[0]

        X = xw[-1].astype(np.float32)          # [N_s, C]
        pred = X @ w                           # [N_s]
        y_true = y.astype(np.float32)          # label_z

        ic = corr_ic(torch.tensor(pred), torch.tensor(y_true))
        ric = rank_ic(torch.tensor(pred), torch.tensor(y_true))

        rows_metrics.append({
            "date": date_str,
            "batch_idx": batch_idx,
            "ic": ic,
            "rank_ic": ric,
            "split": split_id,
        })

        for gidx, p, r in zip(inst_idx, pred.tolist(), y_true.tolist()):
            rows_preds.append({
                "date": date_str,
                "inst_idx": int(gidx),
                "instrument": str(instruments[int(gidx)]),
                "pred": float(p),
                "label": float(r),      # 注意：这是 label_z（训练label），eval_full 会替换成 raw
                "split": split_id,
            })

    return pd.DataFrame(rows_metrics), pd.DataFrame(rows_preds)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, default="")
    p.add_argument("--market", type=str, default="csi300")
    p.add_argument("--npz_path", type=str, default="")
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--window_T", type=int, default=20)
    p.add_argument("--ridge_lambda", type=float, default=1e-2)

    p.add_argument("--run_name", type=str, default="", help="subdir under runs/<market>/")
    args = p.parse_args()

    set_seed(args.seed)

    npz_path = args.npz_path or os.path.join(args.data_dir, f"{args.market}_alpha158_2013-01-01_2023-12-31.npz")
    panel = load_panel_npz(npz_path)

    # 输出目录
    run_dir = os.path.join("runs", args.market, args.run_name) if args.run_name else os.path.join("runs", args.market)
    os.makedirs(run_dir, exist_ok=True)

    # 写配置，方便回溯
    try:
        with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as f:
            json.dump({"args": vars(args), "npz_path": npz_path}, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[WARN] failed to write config.json: {e}")

    all_metrics = []
    all_preds = []

    # 确定特征维度 C
    C = int(panel.features.shape[-1])

    for split_id, (train_rg, valid_rg, test_rg) in enumerate(make_rolling_splits(panel.dates), start=1):
        print(f"\n=== split {split_id}: train={train_rg} valid={valid_rg} test={test_rg} ===")

        train_ds = SlidingWindowDataset(panel, args.window_T, train_rg[0], train_rg[1])
        test_ds = SlidingWindowDataset(panel, args.window_T, test_rg[0], test_rg[1])

        train_loader = DataLoader(train_ds, batch_size=1, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)

        w = fit_ridge_stream(train_loader, C=C, lam=float(args.ridge_lambda))
        metrics_df, preds_df = predict_daily_ridge(w, test_loader, instruments=panel.instruments, split_id=split_id)

        all_metrics.append(metrics_df)
        all_preds.append(preds_df)

    metrics = pd.concat(all_metrics, ignore_index=True)
    preds = pd.concat(all_preds, ignore_index=True)

    m_path = os.path.join(run_dir, "daily_metrics.parquet")
    p_path = os.path.join(run_dir, "daily_preds.parquet")

    metrics.to_parquet(m_path, index=False)
    preds.to_parquet(p_path, index=False)
    preds.to_csv(p_path.replace(".parquet", ".csv"), index=False)

    print(f"Saved metrics to: {m_path}")
    print(f"Saved preds to: {p_path}")


if __name__ == "__main__":
    main()
