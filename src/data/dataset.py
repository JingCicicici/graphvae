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
    # List[pd.Timestamp] 只是一个提示（Typing），意思是“这里装的是一串时间戳”。
    dates: List[pd.Timestamp]
    instruments: List[str]
    # features: [D,N,C] float32
    features: np.ndarray
    # labels:   [D,N]   float32 (next-day return ratio aligned to date)
    labels: np.ndarray
    # Optional metadata for label provenance transparency
    label_source: Optional[str] = None
    label_formula: Optional[str] = None

# 滑动窗口取数据
# 括号里的 (Dataset) 意思是：这个类继承了 PyTorch 官方的 Dataset 模板。
class SlidingWindowDataset(Dataset):
    """Turn a [D,N,C] panel into samples (x_window, y_next) per date.

    For date index s, we return:
      x_window = features[s-T+1 : s+1]  -> [T,N,C]
      y = labels[s]                     -> [N]
    """

    def __init__(self, panel: PanelData, window_T: int, start: int, end: int):
        # assert 是断言，相当于保安。如果传进来的特征不是 3 维的，当场报错赶出去。
        assert panel.features.ndim == 3
        self.panel = panel
        self.T = window_T
        # max(start, window_T - 1)：这是为了防错。
        # 假设从第 0 天开始，但模型要吃 20 天，第 0 天前面根本没有足够的数据！
        # 所以强制要求，起点最少也得是第 19 天（也就是积累够了 20 天数据的那天）。
        self.start = max(start, window_T - 1)
        self.end = min(end, panel.features.shape[0] - 1)
        assert self.start <= self.end

    def __len__(self) -> int:
        return self.end - self.start + 1
    # __getitem__：每次 PyTorch 喊她拿数据，并报出一个编号 idx（比如第 0 顿、第 1 顿），大妈就执行一次这个函数。
    def __getitem__(self, idx: int):
        # 算出当前这顿饭，在真实的 3D 大矩阵里到底是第几天（s 代表 Shifted index）。
        s = self.start + idx
        # xw 就是切出来的这 T 天的数据。它的形状变成了 [天数T, 股票数N, 特征数C]。
        xw = self.panel.features[s - self.T + 1: s + 1]  # [T,N,C] numpy
        # 拿出第 s 天的答案（明天的收益率）
        y = self.panel.labels[s]  # [N] numpy

        # 1) 只要求当天 label 有效（定义 N_s）
        mask = np.isfinite(y)  # [N]
        inst_idx = np.where(mask)[0].astype(np.int64)
        # 第一个冒号 : 意思是时间 T 全保留。
        # 第二个位置放 mask，意思是只留下刚才标记为 True 的股票！停牌的股票全部砍掉！
        # 第三个冒号 : 意思是所有的 158 个特征全保留。
        xw = xw[:, mask, :]  # [T, N_s, C]
        y = y[mask]  # [N_s]

        # 2) 把窗口内缺失特征填 0，保证图构建不会 NaN 崩
        xw = np.nan_to_num(xw, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        y = y.astype(np.float32)
        date_str = str(self.panel.dates[s].date())
        # torch.from_numpy()，这个函数就是把 Numpy 矩阵“无缝包装”成 PyTorch 的 Tensor。
        return (
            torch.from_numpy(xw),  # [T, N_s, C]
            torch.from_numpy(y),  # [N_s]
            date_str,  # str
            torch.from_numpy(inst_idx),  # [N_s] long
        )
# 解开压缩包
def load_panel_npz(path: str) -> PanelData:
    """Load dumped panel created by dump_qlib_alpha158.py"""
    # np.load：Numpy 大哥的解压命令。
    # allow_pickle=True：因为包里存了字符串，必须允许它使用一种叫 pickle 的底层方式来读取。
    obj = np.load(path, allow_pickle=True)
    # obj["dates"] 是把压缩包里的 dates 文件抽出来。
    # .tolist() 是把它从 Numpy 格式变成 Python 的普通列表。
    # [pd.Timestamp(x) for x in ...]：这是一个循环，把里面的纯文本日期（比如 '2013-01-01'）全部转换成专业的 Pandas 时间格式。
    dates = [pd.Timestamp(x) for x in obj["dates"].tolist()]
    instruments = obj["instruments"].tolist()
    # 把特征矩阵抽出来，并强制确认它是 32位小数（float32）。
    features = obj["features"].astype(np.float32)
    labels = obj["labels"].astype(np.float32)
    # 抽说明书。如果压缩包里有，就提取出来；如果没有，就记成 None。
    label_source = obj["label_source"].item() if "label_source" in obj.files else None
    label_formula = obj["label_formula"].item() if "label_formula" in obj.files else None
    # 把刚才抽出来的这 6 样东西，打包塞进 PanelData 储物箱里，然后当做结果交出去！
    return PanelData(dates, instruments, features, labels, label_source=label_source, label_formula=label_formula)
