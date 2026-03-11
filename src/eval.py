from __future__ import annotations

"""Quick evaluation helper.

This script is only for fast sanity checks on precomputed daily returns.
For paper-aligned evaluation (IC/RankIC + TopK-Drop + AR/IR), use `src.eval_full`.
"""

import argparse
import pandas as pd
import torch

from .utils import annualized_return, information_ratio


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pred_path", type=str, default="")
    args = p.parse_args()

    if not args.pred_path:
        print("Quick check only: pass a file with a `daily_ret` column. For paper evaluation use: python -m src.eval_full --pred_path runs/<market>/<run>/daily_preds.parquet")
        return

    df = pd.read_parquet(args.pred_path)
    # expect columns: date, daily_ret
    daily = torch.tensor(df["daily_ret"].values, dtype=torch.float32)
    print("AR:", annualized_return(daily))
    print("IR:", information_ratio(daily))


if __name__ == "__main__":
    main()
