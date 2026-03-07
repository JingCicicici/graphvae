from __future__ import annotations

import argparse
import pandas as pd
import torch

from .utils import annualized_return, information_ratio


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pred_path", type=str, default="")
    args = p.parse_args()

    if not args.pred_path:
        print("This skeleton repo only includes a minimal metric pipeline. Integrate Qlib backtest for full TopK-Drop.")
        return

    df = pd.read_parquet(args.pred_path)
    # expect columns: date, daily_ret
    daily = torch.tensor(df["daily_ret"].values, dtype=torch.float32)
    print("AR:", annualized_return(daily))
    print("IR:", information_ratio(daily))


if __name__ == "__main__":
    main()
