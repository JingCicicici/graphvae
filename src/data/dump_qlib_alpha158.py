from __future__ import annotations

import argparse
import os
from typing import List

import numpy as np
import pandas as pd
from tqdm import tqdm

import qlib
from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandlerLP


def dump_alpha158_panel(provider_uri: str, market: str, out_dir: str,
                        start_time: str = "2013-01-01", end_time: str = "2023-12-31",
                        fit_end_time: str | None = None,
                        drop_bad_days: int = 0) -> str:

    """Dump Alpha158 panel into one npz:
      - features: [D,N,C]
      - labels:   [D,N] (next-day return ratio)
    This uses Qlib's built-in Alpha158 handler (see Qlib docs/examples).

    Note: This script is designed for *reproducible research* rather than max speed.
    """
    qlib.init(provider_uri=provider_uri, region="cn")
    fit_end_time = fit_end_time or end_time
    handler = {
        "class": "Alpha158",
        "module_path": "qlib.contrib.data.handler",
        "kwargs": {
            "start_time": start_time,
            "end_time": end_time,
            "fit_start_time": start_time,
            "fit_end_time": fit_end_time,
            "instruments": market,
        },
    }
    dataset = DatasetH(handler=handler, segments={"all": (start_time, end_time)})

    # dataset.prepare returns MultiIndex DF: index=(datetime, instrument)
    df = dataset.prepare("all", col_set=["feature", "label"], data_key=DataHandlerLP.DK_L)

    # Separate feature and label, align by date/instrument
    feat = df["feature"]
    label = df["label"]

    dates = sorted(feat.index.get_level_values("datetime").unique())
    instruments = sorted(feat.index.get_level_values("instrument").unique())

    # Feature columns are like feature_0 ... feature_157
    feat_cols = list(feat.columns)
    C = len(feat_cols)
    D = len(dates)
    N = len(instruments)

    # Build dense arrays
    X = np.full((D, N, C), np.nan, dtype=np.float32)
    Y = np.full((D, N), np.nan, dtype=np.float32)

    date_to_i = {d: i for i, d in enumerate(dates)}
    inst_to_j = {inst: j for j, inst in enumerate(instruments)}

    # Fill
    for (dt, inst), row in tqdm(feat.iterrows(), total=len(feat), desc="Filling features"):
        i = date_to_i[dt]
        j = inst_to_j[inst]
        X[i, j, :] = row.values.astype(np.float32)

    for (dt, inst), val in tqdm(label.iloc[:, 0].items(), total=len(label), desc="Filling labels"):
        i = date_to_i[dt]
        j = inst_to_j[inst]
        Y[i, j] = float(val)

    # Drop dates with too many NaNs (robust for dynamic constituents & short data coverage)
    if drop_bad_days:
        finite_cnt = np.isfinite(Y).sum(axis=1)
        max_valid = int(finite_cnt.max()) if finite_cnt.size else 0
        min_valid = 30 if max_valid == 0 else max(30, int(0.25 * max_valid))
        valid_date_mask = finite_cnt >= min_valid

        print(
            f"[dump] N={N}, max_valid={max_valid}, min_valid={min_valid}, kept_days={int(valid_date_mask.sum())}/{len(valid_date_mask)}")

        X = X[valid_date_mask]
        Y = Y[valid_date_mask]
        dates = [d for d, m in zip(dates, valid_date_mask.tolist()) if m]

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{market}_alpha158_{start_time}_{end_time}.npz")
    np.savez_compressed(
        out_path,
        dates=np.array([str(d) for d in dates], dtype=object),
        instruments=np.array(instruments, dtype=object),
        features=X,
        labels=Y,
        feat_cols=np.array(feat_cols, dtype=object),
        # Keep label provenance explicit for auditability.
        label_source=np.array("qlib.contrib.data.handler.Alpha158::label", dtype=object),
        label_formula=np.array("handler_config_dependent (commonly Ref($close, -2)/Ref($close, -1)-1)", dtype=object),
    )
    return out_path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--provider_uri", type=str, required=True)
    p.add_argument("--market", type=str, default="csi300", help="csi300/csi500/csi1000 or a Qlib instrument set")
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--start_time", type=str, default="2013-01-01")
    p.add_argument("--end_time", type=str, default="2023-12-31")
    p.add_argument("--fit_end_time", type=str, default=None,
                   help="end date used to fit Qlib processors (to avoid look-ahead). If None, use end_time.")
    p.add_argument("--drop_bad_days", type=int, default=0,
                   help="1: drop days with too few valid labels; 0: keep all days (recommended)")
    args = p.parse_args()

    out = dump_alpha158_panel(
        args.provider_uri, args.market, args.out_dir,
        args.start_time, args.end_time,
        fit_end_time=args.fit_end_time,
        drop_bad_days=args.drop_bad_days
    )

    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
