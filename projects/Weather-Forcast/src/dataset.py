"""
dataset.py — PyTorch Dataset 封装
===================================
从 .npz 文件加载滑动窗口数据，供 DataLoader 使用。
每个样本返回 (X, y) 元组，形状:
    X: (window_len, n_features)
    y: 标量 (归一化 target)

用法:
    from dataset import WeatherDataset
    ds = WeatherDataset("path/to/train.npz")
    loader = DataLoader(ds, batch_size=64, shuffle=True)
"""

import numpy as np
import torch
from torch.utils.data import Dataset


class WeatherDataset(Dataset):
    def __init__(self, npz_path):
        data = np.load(npz_path)
        self.X = torch.from_numpy(data["X"].astype(np.float32))
        self.y = torch.from_numpy(data["y"].astype(np.float32))

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
