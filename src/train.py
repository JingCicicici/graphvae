from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from typing import Tuple, Optional

import pandas as pd
import torch
from torch.utils.data import DataLoader

from .models.graphvae import GraphVAE, GraphVAEConfig
from .data.dataset import load_panel_npz, SlidingWindowDataset
from .utils import set_seed, corr_ic, rank_ic
from .models.graph_builder import GraphBuildConfig, build_neighbors_from_window


def make_rolling_splits(dates, train_years=5, valid_years=1, test_years=1):
    """Yield tuples of (train_idx_range, valid_idx_range, test_idx_range) by year.

    Simplified rolling split builder.
    """
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


def train_one_split(
    model: GraphVAE,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    device: torch.device,
    lr: float = 1e-3,
    epochs: int = 10,
) -> Tuple[GraphVAE, Optional[dict], float]:
    """Train one rolling split, return (best_model_loaded, best_state_dict, best_valid_loss)."""
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best = float("inf")
    best_state = None

    for ep in range(1, epochs + 1):
        # -------- train --------
        model.train()
        for xw, y, _, _ in train_loader:   # dataset returns 4 values
            xw = xw.squeeze(0).to(device)  # [T, N_s, C]
            y = y.squeeze(0).to(device)    # [N_s]

            out = model(xw, y=y)
            loss = out["loss"]

            opt.zero_grad()
            loss.backward()
            opt.step()

        # -------- valid --------
        model.eval()
        vloss = 0.0
        n = 0
        with torch.no_grad():
            for xw, y, _, _ in valid_loader:
                xw = xw.squeeze(0).to(device)
                y = y.squeeze(0).to(device)

                out = model(xw, y=y)
                loss = out["loss"]

                vloss += float(loss.item())
                n += 1

        vloss /= max(n, 1)

        if vloss < best:
            best = vloss
            # keep a CPU copy to save memory
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

        print(f"epoch={ep:03d} valid_loss={vloss:.6f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_state, float(best)


def predict_daily(model, loader, device, instruments=None, split_id=None):
    """Return (metrics_df, preds_df)."""
    model.eval()
    rows_metrics = []
    rows_preds = []

    with torch.no_grad():
        for batch_idx, (xw, y, date_str, inst_idx) in enumerate(loader):
            xw = xw.squeeze(0).to(device)  # [T, N_s, C]
            y = y.squeeze(0).to(device)    # [N_s]

            if isinstance(inst_idx, torch.Tensor):
                inst_idx = inst_idx.squeeze(0)

            out = model(xw, y=None)
            mu_pred = out["mu_pred"].detach().cpu().flatten()
            y_true = y.detach().cpu().flatten()

            if isinstance(date_str, (list, tuple)):
                date_str = date_str[0]

            # ---- daily metrics ----
            ic = corr_ic(mu_pred, y_true)
            ric = rank_ic(mu_pred, y_true)
            rows_metrics.append({
                "date": date_str,
                "batch_idx": batch_idx,
                "ic": ic,
                "rank_ic": ric,
                "split": split_id if split_id is not None else -1,
            })

            # ---- per-stock preds ----
            inst_idx_list = inst_idx.detach().cpu().tolist() if isinstance(inst_idx, torch.Tensor) else list(inst_idx)
            pred_list = mu_pred.tolist()
            label_list = y_true.tolist()

            if not (len(inst_idx_list) == len(pred_list) == len(label_list)):
                raise ValueError(
                    f"Length mismatch: inst_idx={len(inst_idx_list)} pred={len(pred_list)} label={len(label_list)}"
                )

            for gidx, p, r in zip(inst_idx_list, pred_list, label_list):
                row = {
                    "date": date_str,
                    "inst_idx": int(gidx),
                    "pred": float(p),
                    "label": float(r),
                    "split": split_id if split_id is not None else -1,
                }
                if instruments is not None:
                    row["instrument"] = str(instruments[int(gidx)])
                rows_preds.append(row)

    metrics_df = pd.DataFrame(rows_metrics)
    preds_df = pd.DataFrame(rows_preds)
    return metrics_df, preds_df


def smoke_test():
    T, N, C = 20, 64, 158
    x = torch.randn(T, N, C)
    y = torch.randn(N)
    cfg = GraphVAEConfig(window_T=T, num_features=C)
    model = GraphVAE(cfg)
    out = model(x, y=y)
    print("smoke loss:", out["loss"].item())


def graph_test(npz_path: str):
    panel = load_panel_npz(npz_path)
    cfg = GraphVAEConfig()
    ds = SlidingWindowDataset(panel, cfg.window_T, 0, min(200, len(panel.dates) - 1))
    xw, y, date_str, inst_idx = ds[0]
    xw = xw.float()
    neighbors = build_neighbors_from_window(xw, GraphBuildConfig(cfg.eta_quantile, cfg.alpha_decay, cfg.topk))
    print("date:", date_str)
    print("xw shape:", tuple(xw.shape))
    print("y shape:", tuple(y.shape))
    print("inst_idx len:", int(inst_idx.numel()))
    print("neighbors shape:", tuple(neighbors.shape))
    print("neighbors[0][:10]:", neighbors[0][:10].tolist())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--smoke_test", type=int, default=0)
    p.add_argument("--data_dir", type=str, default="")
    p.add_argument("--market", type=str, default="csi300")
    p.add_argument("--npz_path", type=str, default="")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--graph_test", type=int, default=0)

    # ✅ 方案2：实验目录隔离
    p.add_argument("--run_name", type=str, default="", help="subdir under runs/<market>/ to avoid overwriting")
    # ✅ 可选：保存每个 split 的 best 权重
    p.add_argument("--save_ckpt", type=int, default=1, help="1 to save best checkpoint per split")

    args = p.parse_args()

    if args.graph_test:
        graph_test(args.npz_path or os.path.join(args.data_dir, f"{args.market}_alpha158_2013-01-01_2023-12-31.npz"))
        return

    if args.smoke_test:
        smoke_test()
        return

    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    npz_path = args.npz_path or os.path.join(args.data_dir, f"{args.market}_alpha158_2013-01-01_2023-12-31.npz")
    panel = load_panel_npz(npz_path)

    cfg = GraphVAEConfig()

    # ✅ 输出目录：runs/<market>/<run_name>（不传 run_name 就回到 runs/<market>）
    run_dir = os.path.join("runs", args.market, args.run_name) if args.run_name else os.path.join("runs", args.market)
    os.makedirs(run_dir, exist_ok=True)

    # 记录本次实验配置，方便回溯
    try:
        with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as f:
            json.dump({"args": vars(args), "model_cfg": asdict(cfg), "npz_path": npz_path}, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[WARN] failed to write config.json: {e}")

    all_results = []
    all_preds = []

    for split_id, (train_rg, valid_rg, test_rg) in enumerate(make_rolling_splits(panel.dates), start=1):
        print(f"\n=== split {split_id}: train={train_rg} valid={valid_rg} test={test_rg} ===")

        model = GraphVAE(cfg).to(device)

        train_ds = SlidingWindowDataset(panel, cfg.window_T, train_rg[0], train_rg[1])
        valid_ds = SlidingWindowDataset(panel, cfg.window_T, valid_rg[0], valid_rg[1])
        test_ds = SlidingWindowDataset(panel, cfg.window_T, test_rg[0], test_rg[1])

        train_loader = DataLoader(train_ds, batch_size=1, shuffle=True)
        valid_loader = DataLoader(valid_ds, batch_size=1, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)

        model, best_state, best_vloss = train_one_split(
            model, train_loader, valid_loader, device, lr=args.lr, epochs=args.epochs
        )

        # ✅ 保存 best checkpoint（每个 split 一个）
        if args.save_ckpt:
            ckpt_path = os.path.join(run_dir, f"split{split_id:02d}_best.pt")
            payload = {
                "split_id": split_id,
                "best_valid_loss": best_vloss,
                "model_state_dict": model.state_dict(),  # 已经 load 了 best
                "model_cfg": asdict(cfg),
                "args": vars(args),
                "npz_path": npz_path,
                "train_range": train_rg,
                "valid_range": valid_rg,
                "test_range": test_rg,
            }
            torch.save(payload, ckpt_path)
            print(f"Saved ckpt to: {ckpt_path}")

        metrics_df, preds_df = predict_daily(
            model, test_loader, device,
            instruments=panel.instruments,
            split_id=split_id
        )
        all_results.append(metrics_df)
        all_preds.append(preds_df)

    # ---- save metrics ----
    out_df = pd.concat(all_results, ignore_index=True)
    out_path = os.path.join(run_dir, "daily_metrics.parquet")
    try:
        out_df.to_parquet(out_path, index=False)
        print(f"Saved metrics to: {out_path}")
    except Exception as e:
        out_csv = out_path.replace(".parquet", ".csv")
        out_df.to_csv(out_csv, index=False)
        print(f"[WARN] save parquet failed ({e}). Saved CSV to: {out_csv}")

    # ---- save preds ----
    pred_df = pd.concat(all_preds, ignore_index=True)
    pred_path = os.path.join(run_dir, "daily_preds.parquet")
    try:
        pred_df.to_parquet(pred_path, index=False)
        print(f"Saved preds to: {pred_path}")
    except Exception as e:
        pred_csv = pred_path.replace(".parquet", ".csv")
        pred_df.to_csv(pred_csv, index=False)
        print(f"[WARN] save parquet failed ({e}). Saved CSV to: {pred_csv}")

    # 可选：同时导出一份 CSV 方便快速查看（不会影响 parquet）
    try:
        pred_df.to_csv(pred_path.replace(".parquet", ".csv"), index=False)
    except Exception:
        pass


if __name__ == "__main__":
    main()