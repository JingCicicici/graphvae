from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .models.graphvae import GraphVAE, GraphVAEConfig
from .data.dataset import load_panel_npz, SlidingWindowDataset, PanelData
from .utils import set_seed, corr_ic, rank_ic


def make_rolling_splits(dates, market: str = "csi300", train_years=None, valid_years=1, test_years=1):
    if train_years is None:
        train_years = 3 if str(market).lower() == "csi1000" else 5

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


def attach_raw_labels_from_qlib(panel: PanelData, qlib_uri: str) -> np.ndarray:
    """Return labels_raw [D,N] = close_{t+1}/close_t - 1 aligned to panel.dates & panel.instruments."""
    import qlib
    from qlib.data import D

    qlib.init(provider_uri=qlib_uri, region="cn")

    dates = pd.to_datetime(panel.dates)
    start = str(dates.min().date())
    end = str((dates.max() + pd.Timedelta(days=20)).date())

    insts = list(panel.instruments)

    close = D.features(insts, ["$close"], start_time=start, end_time=end)
    close = close.reset_index().rename(columns={"datetime": "date"})
    close["date"] = pd.to_datetime(close["date"])

    # pivot to [date x instrument]
    pv = close.pivot(index="date", columns="instrument", values="$close").sort_index()
    pv = pv.reindex(columns=insts)              # align instrument order
    pv = pv.reindex(dates)                      # align date index to panel dates

    ret = pv.shift(-1) / pv - 1.0               # next-day return
    labels_raw = ret.to_numpy(dtype=np.float32) # [D,N]
    return labels_raw


def train_one_split(model, train_loader, valid_loader, device, lr=1e-3, epochs=50):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best = float("inf")
    best_state = None

    for ep in range(1, epochs + 1):
        model.train()
        for xw, y, _, _ in train_loader:
            xw = xw.squeeze(0).to(device)  # [T,N,C]
            y = y.squeeze(0).to(device)    # [N]
            out = model(xw, y=y)
            loss = out["loss"]
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
                out = model(xw, y=y)
                vloss += float(out["loss"].item())
                n += 1
        vloss /= max(n, 1)

        if vloss < best:
            best = vloss
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

        print(f"epoch={ep:03d} valid_loss={vloss:.6f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_state, float(best)


def predict_daily(model, loader, device, instruments=None, split_id=None):
    model.eval()
    rows_metrics, rows_preds = [], []

    with torch.no_grad():
        for batch_idx, (xw, y, date_str, inst_idx) in enumerate(loader):
            xw = xw.squeeze(0).to(device)  # [T,N,C]
            y = y.squeeze(0).to(device)    # [N]

            if isinstance(inst_idx, torch.Tensor):
                inst_idx = inst_idx.squeeze(0)

            out = model(xw, y=None)
            mu_pred = out["mu_pred"].detach().cpu().flatten()
            y_true = y.detach().cpu().flatten()

            if isinstance(date_str, (list, tuple)):
                date_str = date_str[0]

            ic = corr_ic(mu_pred, y_true)
            ric = rank_ic(mu_pred, y_true)
            rows_metrics.append({"date": date_str, "batch_idx": batch_idx, "ic": ic, "rank_ic": ric, "split": split_id or -1})

            idx_list = inst_idx.detach().cpu().tolist()
            for gidx, p, r in zip(idx_list, mu_pred.tolist(), y_true.tolist()):
                row = {
                    "date": date_str,
                    "inst_idx": int(gidx),
                    "pred": float(p),
                    "label": float(r),   # 注意：这里已经是 raw y
                    "split": split_id or -1,
                }
                if instruments is not None:
                    row["instrument"] = str(instruments[int(gidx)])
                rows_preds.append(row)

    return pd.DataFrame(rows_metrics), pd.DataFrame(rows_preds)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, default="")
    p.add_argument("--market", type=str, default="csi300")
    p.add_argument("--npz_path", type=str, default="")
    p.add_argument("--qlib_uri", type=str, default="/root/autodl-tmp/qlib_data")

    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda")

    # allow quick tuning (paper defaults)
    p.add_argument("--hidden_dim", type=int, default=64)
    p.add_argument("--factor_dim", type=int, default=16)
    p.add_argument("--eta", type=float, default=90.0)
    p.add_argument("--alpha", type=float, default=0.9)
    p.add_argument("--topk", type=int, default=5)
    p.add_argument("--kl_weight", type=float, default=1.0)

    p.add_argument("--run_name", type=str, default="")
    p.add_argument("--save_ckpt", type=int, default=1)

    args = p.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    npz_path = args.npz_path or os.path.join(args.data_dir, f"{args.market}_alpha158_2013-01-01_2023-12-31.npz")
    panel = load_panel_npz(npz_path)

    if panel.label_source or panel.label_formula:
        print(f"[label] source={panel.label_source} formula={panel.label_formula}")
    else:
        print("[label][WARN] label metadata missing in npz; please verify label definition manually.")

    # build raw labels aligned to panel
    labels_raw = attach_raw_labels_from_qlib(panel, args.qlib_uri)
    panel_raw = PanelData(panel.dates, panel.instruments, panel.features, labels_raw)

    # IMPORTANT: next-day return has no label on the last date; cap max index
    max_label_idx = len(panel_raw.dates) - 2
    print(f"[data] max_label_idx={max_label_idx} (last usable date for next-day label)")

    cfg = GraphVAEConfig(
        window_T=20,
        num_features=panel_raw.features.shape[-1],
        hidden_dim=args.hidden_dim,
        factor_dim=args.factor_dim,
        eta_quantile=args.eta,
        alpha_decay=args.alpha,
        topk=args.topk,
        kl_weight=args.kl_weight,
    )

    run_dir = os.path.join("runs", args.market, args.run_name) if args.run_name else os.path.join("runs", args.market)
    os.makedirs(run_dir, exist_ok=True)

    # save config
    try:
        with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as f:
            json.dump({"args": vars(args), "model_cfg": asdict(cfg), "npz_path": npz_path}, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

    all_metrics, all_preds = [], []

    for split_id, (train_rg, valid_rg, test_rg) in enumerate(make_rolling_splits(panel_raw.dates, market=args.market), start=1):
        # cap end to max_label_idx
        train_end = min(train_rg[1], max_label_idx)
        valid_end = min(valid_rg[1], max_label_idx)
        test_end  = min(test_rg[1],  max_label_idx)
        if train_end < train_rg[0] or valid_end < valid_rg[0] or test_end < test_rg[0]:
            continue

        print(
            f"\n=== split {split_id}: raw train={train_rg} valid={valid_rg} test={test_rg} | "
            f"capped train=({train_rg[0]}, {train_end}) valid=({valid_rg[0]}, {valid_end}) test=({test_rg[0]}, {test_end}) ==="
        )

        model = GraphVAE(cfg).to(device)

        train_ds = SlidingWindowDataset(panel_raw, cfg.window_T, train_rg[0], train_end)
        valid_ds = SlidingWindowDataset(panel_raw, cfg.window_T, valid_rg[0], valid_end)
        test_ds  = SlidingWindowDataset(panel_raw, cfg.window_T, test_rg[0],  test_end)

        train_loader = DataLoader(train_ds, batch_size=1, shuffle=True)
        valid_loader = DataLoader(valid_ds, batch_size=1, shuffle=False)
        test_loader  = DataLoader(test_ds,  batch_size=1, shuffle=False)

        model, best_state, best_vloss = train_one_split(model, train_loader, valid_loader, device, lr=args.lr, epochs=args.epochs)

        if args.save_ckpt:
            ckpt_path = os.path.join(run_dir, f"split{split_id:02d}_best.pt")
            torch.save({"model_state_dict": model.state_dict(), "best_valid_loss": best_vloss, "cfg": asdict(cfg)}, ckpt_path)
            print(f"Saved ckpt to: {ckpt_path}")

        m_df, p_df = predict_daily(model, test_loader, device, instruments=panel_raw.instruments, split_id=split_id)
        all_metrics.append(m_df)
        all_preds.append(p_df)

    if not all_metrics:
        raise RuntimeError("No valid rolling split after boundary checks; please verify date range and label availability.")

    metrics = pd.concat(all_metrics, ignore_index=True)
    preds   = pd.concat(all_preds, ignore_index=True)

    metrics.to_parquet(os.path.join(run_dir, "daily_metrics.parquet"), index=False)
    preds.to_parquet(os.path.join(run_dir, "daily_preds.parquet"), index=False)
    preds.to_csv(os.path.join(run_dir, "daily_preds.csv"), index=False)

    print(f"Saved metrics to: {os.path.join(run_dir, 'daily_metrics.parquet')}")
    print(f"Saved preds to: {os.path.join(run_dir, 'daily_preds.parquet')}")


if __name__ == "__main__":
    main()
