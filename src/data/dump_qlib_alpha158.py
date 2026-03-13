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
    # 这个参数分别表示运行时的输入参数：Qlib 原始数据文件夹在哪，跑哪个股票池等

    """Dump Alpha158 panel into one npz:
      - features: [D,N,C]
      - labels:   [D,N] (next-day return ratio)
    This uses Qlib's built-in Alpha158 handler (see Qlib docs/examples).

    Note: This script is designed for *reproducible research* rather than max speed.
    """
    qlib.init(provider_uri=provider_uri, region="cn")
    fit_end_time = fit_end_time or end_time
    # handler 字典：这是传给 Qlib 的“配方表”。
    # 意思是：我要用 "Alpha158" 这个计算模块，去算 "market" 里这些股票在这个时间段的 158 个特征。
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
    # DatasetH: Qlib 里的高级数据集管理类。它会根据上面的配方表去准备数据。
    dataset = DatasetH(handler=handler, segments={"all": (start_time, end_time)})

    # dataset.prepare: 正式让 Qlib 开始计算！
    # "all": 取全部时间段。
    # col_set=["feature", "label"]: 告诉它我既要特征（158个因子），也要标签（明天的收益率）。
    # DataHandlerLP.DK_L: 这是一个底层常量，指代 "Data Key Local"，意思就是把所有算好的全量数据拉出来。
    # df (DataFrame): 返回的是一个带有双重索引 (datetime, instrument) 的大表格。
    df = dataset.prepare("all", col_set=["feature", "label"], data_key=DataHandlerLP.DK_L)

    # df["feature"] / df["label"]: 把大表格切成两块，feat 装特征，label 装标签。
    feat = df["feature"]
    label = df["label"]
    # feat.index.get_level_values("datetime"): 从双重索引里，把名字叫 "datetime" 的那一层扒出来。
    # .unique(): 去重。因为一天有300只股票，日期会重复300次，去重后就是纯粹的交易日历。
    dates = sorted(feat.index.get_level_values("datetime").unique())
    instruments = sorted(feat.index.get_level_values("instrument").unique())

    # feat.columns: 获取特征表的所有列名（比如 feature_0 到 feature_157）。
    feat_cols = list(feat.columns)
    C = len(feat_cols)
    D = len(dates)
    N = len(instruments)

    # np.full(形状, 填充物, 数据类型): 造一个三维大空壳 X 和二维大空壳 Y。
    # np.nan: Not a Number（空值）。先全填上空值，等下往里塞真实数据。
    # dtype=np.float32: 为了给显卡省内存，强制用 32 位浮点数（默认是64位太占空间）。
    X = np.full((D, N, C), np.nan, dtype=np.float32)
    Y = np.full((D, N), np.nan, dtype=np.float32)
    # enumerate将日期变成数值对
    # d: i：意思是把这对数据，按照 日期: 序号 的格式存进字典里，for i, d in enumerate(dates)：意思是每次拿出一对组合。i 就是序号 ，d 就是日期 
    date_to_i = {d: i for i, d in enumerate(dates)}
    inst_to_j = {inst: j for j, inst in enumerate(instruments)}

    # feat.iterrows(): 每次吐出一行数据。dt 是日期，inst 是股票代码，row 是那一行的158个特征值。
    for (dt, inst), row in tqdm(feat.iterrows(), total=len(feat), desc="Filling features"):
        i = date_to_i[dt]# 查出该填在第几个日子
        j = inst_to_j[inst]# 查出该填在第几只股票
        X[i, j, :] = row.values.astype(np.float32)#":" 代表这158个特征全塞进这个坐标里。
        
    # label.iloc[:, 0].items(): iloc[:, 0] 表示取标签表的第一列，items() 挨个吐出 (索引, 值)。
    for (dt, inst), val in tqdm(label.iloc[:, 0].items(), total=len(label), desc="Filling labels"):
        i = date_to_i[dt]
        j = inst_to_j[inst]
        Y[i, j] = float(val)# 把明天的收益率填进坐标

    # np.isfinite(Y): 判断 Y 里面的值是不是正常的数字（把 nan 和无限大剔除），返回一堆 True/False。
        # .sum(axis=1): 按行（也就是按天）加总。算出每天有几只股票的收益率是正常的。
    if drop_bad_days:
        finite_cnt = np.isfinite(Y).sum(axis=1)
        # 算出一个及格线 (min_valid)，每天起码得有那么多只股票正常，这天才算数
        max_valid = int(finite_cnt.max()) if finite_cnt.size else 0
        min_valid = 30 if max_valid == 0 else max(30, int(0.25 * max_valid))
        valid_date_mask = finite_cnt >= min_valid

        print(
            f"[dump] N={N}, max_valid={max_valid}, min_valid={min_valid}, kept_days={int(valid_date_mask.sum())}/{len(valid_date_mask)}")
        # X[valid_date_mask]: numpy 的高级过滤魔法，直接把 False 对应的那几天全部删掉！
        X = X[valid_date_mask]
        Y = Y[valid_date_mask]
        dates = [d for d, m in zip(dates, valid_date_mask.tolist()) if m]
    # os.makedirs(..., exist_ok=True): 建个文件夹，如果已经存在了别报错 (exist_ok)。
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{market}_alpha158_{start_time}_{end_time}.npz")
    # 压缩文件
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
