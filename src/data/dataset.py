from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


@dataclass
class PanelData:
    dates: List[pd.Timestamp]
    instruments: List[str]
    # features: [D,N,C] float32
    features: np.ndarray
    # labels:   [D,N]   float32 (next-day return ratio aligned to date)
    labels: np.ndarray


class SlidingWindowDataset(Dataset):
    """Turn a [D,N,C] panel into samples (x_window, y_next) per date.

    For date index s, we return:
      x_window = features[s-T+1 : s+1]  -> [T,N,C]
      y = labels[s]                     -> [N]
    """

    def __init__(self, panel: PanelData, window_T: int, start: int, end: int):
        assert panel.features.ndim == 3
        self.panel = panel
        self.T = window_T
        self.start = max(start, window_T - 1)
        self.end = min(end, panel.features.shape[0] - 1)
        assert self.start <= self.end

    def __len__(self) -> int:
        return self.end - self.start + 1

    def __getitem__(self, idx: int):
        s = self.start + idx
        xw = self.panel.features[s - self.T + 1: s + 1]  # [T,N,C] numpy
        y = self.panel.labels[s]  # [N] numpy

        # 1) 只要求当天 label 有效（定义 N_s）
        mask = np.isfinite(y)  # [N]
        inst_idx = np.where(mask)[0].astype(np.int64)
        xw = xw[:, mask, :]  # [T, N_s, C]
        y = y[mask]  # [N_s]

        # 2) 把窗口内缺失特征填 0，保证图构建不会 NaN 崩
        xw = np.nan_to_num(xw, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        y = y.astype(np.float32)
        date_str = str(self.panel.dates[s].date())
        return (
            torch.from_numpy(xw),  # [T, N_s, C]
            torch.from_numpy(y),  # [N_s]
            date_str,  # str
            torch.from_numpy(inst_idx),  # [N_s] long
        )

def load_panel_npz(path: str) -> PanelData:
    """Load dumped panel created by dump_qlib_alpha158.py"""
    obj = np.load(path, allow_pickle=True)
    dates = [pd.Timestamp(x) for x in obj["dates"].tolist()]
    instruments = obj["instruments"].tolist()
    features = obj["features"].astype(np.float32)
    labels = obj["labels"].astype(np.float32)
    return PanelData(dates, instruments, features, labels)
