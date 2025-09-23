# -*- coding: utf-8 -*-
import datetime
import random
from typing import Tuple
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Sampler
import math

from ts_benchmark.baselines.time_series_library.utils.timefeatures import (
    time_features,
)
from ts_benchmark.utils.data_processing import split_before


class SlidingWindowDataLoader:
    """
    SlidingWindDataLoader class.

    This class encapsulates a sliding window data loader for generating time series training samples.
    """

    def __init__(
        self,
        dataset: pd.DataFrame,
        batch_size: int = 1,
        history_length: int = 10,
        prediction_length: int = 2,
        shuffle: bool = True,
    ):
        """
        Initialize SlidingWindDataLoader.

        :param dataset: Pandas DataFrame containing time series data.
        :param batch_size: Batch size.
        :param history_length: The length of historical data.
        :param prediction_length: The length of the predicted data.
        :param shuffle: Whether to shuffle the dataset.
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.history_length = history_length
        self.prediction_length = prediction_length
        self.shuffle = shuffle
        self.current_index = 0

    def __len__(self) -> int:
        """
        Returns the length of the data loader.

        :return: The length of the data loader.
        """
        return len(self.dataset) - self.history_length - self.prediction_length + 1

    def __iter__(self) -> "SlidingWindowDataLoader":
        """
        Create an iterator and return.

        :return: Data loader iterator.
        """
        if self.shuffle:
            self._shuffle_dataset()
        self.current_index = 0
        return self

    def __next__(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate data for the next batch.

        :return: A tuple containing input data and target data.
        """
        if self.current_index >= len(self):
            raise StopIteration

        batch_inputs = []
        batch_targets = []
        for _ in range(self.batch_size):
            window_data = self.dataset.iloc[
                self.current_index : self.current_index
                + self.history_length
                + self.prediction_length,
                :,
            ]
            if len(window_data) < self.history_length + self.prediction_length:
                raise StopIteration  # Stop iteration when the dataset is less than one window size and prediction step size

            inputs = window_data.iloc[: self.history_length].values
            targets = window_data.iloc[
                self.history_length : self.history_length + self.prediction_length
            ].values

            batch_inputs.append(inputs)
            batch_targets.append(targets)
            self.current_index += 1

        # Convert NumPy array to PyTorch tensor
        batch_inputs = torch.tensor(batch_inputs, dtype=torch.float32)
        batch_targets = torch.tensor(batch_targets, dtype=torch.float32)

        return batch_inputs, batch_targets

    def _shuffle_dataset(self):
        """
        Shuffle the dataset.
        """
        self.dataset = self.dataset.sample(frac=1).reset_index(drop=True)


def train_val_split(train_data, ratio, seq_len):
    if ratio == 1:
        return train_data, None

    elif seq_len is not None:
        border = int((train_data.shape[0]) * ratio)

        train_data_value, valid_data_rest = split_before(train_data, border)
        train_data_rest, valid_data = split_before(train_data, border - seq_len)
        return train_data_value, valid_data
    else:
        border = int((train_data.shape[0]) * ratio)

        train_data_value, valid_data_rest = split_before(train_data, border)
        return train_data_value, valid_data_rest


def decompose_time(
    time: np.ndarray,
    freq: str,
) -> np.ndarray:
    """
    Split the given array of timestamps into components based on the frequency.

    :param time: Array of timestamps.
    :param freq: The frequency of the time stamp.
    :return: Array of timestamp components.
    """
    df_stamp = pd.DataFrame(pd.to_datetime(time), columns=["date"])
    freq_scores = {
        "m": 0,
        "w": 1,
        "b": 2,
        "d": 2,
        "h": 3,
        "t": 4,
        "s": 5,
    }
    max_score = max(freq_scores.values())
    df_stamp["month"] = df_stamp.date.dt.month
    if freq_scores.get(freq, max_score) >= 1:
        df_stamp["day"] = df_stamp.date.dt.day
    if freq_scores.get(freq, max_score) >= 2:
        df_stamp["weekday"] = df_stamp.date.dt.weekday
    if freq_scores.get(freq, max_score) >= 3:
        df_stamp["hour"] = df_stamp.date.dt.hour
    if freq_scores.get(freq, max_score) >= 4:
        df_stamp["minute"] = df_stamp.date.dt.minute
    if freq_scores.get(freq, max_score) >= 5:
        df_stamp["second"] = df_stamp.date.dt.second
    return df_stamp.drop(["date"], axis=1).values


def get_time_mark(
    time_stamp: np.ndarray,
    timeenc: int,
    freq: str,
) -> np.ndarray:
    """
    Extract temporal features from the time stamp.

    :param time_stamp: The time stamp ndarray.
    :param timeenc: The time encoding type.
    :param freq: The frequency of the time stamp.
    :return: The mark of the time stamp.
    """
    if timeenc == 0:
        origin_size = time_stamp.shape
        data_stamp = decompose_time(time_stamp.flatten(), freq)
        data_stamp = data_stamp.reshape(origin_size + (-1,))
    elif timeenc == 1:
        origin_size = time_stamp.shape
        data_stamp = time_features(pd.to_datetime(time_stamp.flatten()), freq=freq)
        data_stamp = data_stamp.transpose(1, 0)
        data_stamp = data_stamp.reshape(origin_size + (-1,))
    else:
        raise ValueError("Unknown time encoding {}".format(timeenc))
    return data_stamp.astype(np.float32)


def forecasting_data_provider(data, config, timeenc, batch_size, shuffle, drop_last):
    dataset = DatasetForTransformer(
        dataset=data,
        history_len=config.seq_len,
        prediction_len=config.horizon,
        label_len=config.label_len,
        timeenc=timeenc,
        freq=config.freq,
    )
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        drop_last=drop_last,
    )

    return dataset, data_loader


class DatasetForTransformer:
    def __init__(
        self,
        dataset: pd.DataFrame,
        history_len: int = 10,
        prediction_len: int = 2,
        label_len: int = 5,
        timeenc: int = 1,
        freq: str = "h",
    ):
        # init

        self.dataset = dataset
        self.history_length = history_len
        self.prediction_length = prediction_len
        self.label_length = label_len
        self.current_index = 0
        self.timeenc = timeenc
        self.freq = freq
        self.__read_data__()

    def __len__(self) -> int:
        """
        Returns the length of the data loader.

        :return: The length of the data loader.
        """
        return len(self.dataset) - self.history_length - self.prediction_length + 1

    def __read_data__(self):
        df_stamp = self.dataset.reset_index()
        df_stamp = df_stamp[["date"]].values.transpose(1, 0)
        data_stamp = get_time_mark(df_stamp, self.timeenc, self.freq)[0]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.history_length
        r_begin = s_end - self.label_length
        r_end = r_begin + self.label_length + self.prediction_length

        seq_x = self.dataset[s_begin:s_end]
        seq_y = self.dataset[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        seq_x = torch.tensor(seq_x.values, dtype=torch.float32)
        seq_y = torch.tensor(seq_y.values, dtype=torch.float32)
        seq_x_mark = torch.tensor(seq_x_mark, dtype=torch.float32)
        seq_y_mark = torch.tensor(seq_y_mark, dtype=torch.float32)
        return seq_x, seq_y, seq_x_mark, seq_y_mark


import numpy as np
import torch
from torch.utils.data import DataLoader

import numpy as np
import torch
from torch.utils.data import DataLoader

# -------------------------------------------------------------
class SegLoader(object):
    """
    • 连续   → float32
    • 离散   → 显式 One-Hot (0/1, int64)  ➟ 后续共享 nn.Embedding(2, d_model)
    • 样本下标、采样密度、步长  fully 兼容你最初那段逻辑
    """
    def __init__(self,
                 data,               # pandas.DataFrame 或 ndarray
                 win_size: int,
                 step: int,
                 mode: str = "train",
                 sampling_rate: float = 1.0,
                 discrete=False,
                 discrete_cols = None,
                 continuous_cols = None):
        self.mode = mode
        self.sampling_rate = sampling_rate
        self.step = step
        self.win_size = win_size

        # ---------- 原始数据 & 标签 ----------
        # 若 test_labels 与 data 不同，可在此处传入外部参数。
        self.data = data   # 为防止花式索引
        self.test_labels = self.data.copy()

        self.discrete = discrete
        if self.discrete:
            if discrete_cols is None or continuous_cols is None:
                # ---------- 连续 / 离散列划分 ----------
                self.continuous_cols, self.discrete_cols, self.constant_cols = [], [], []
                self.discrete_nums = []  
                for col in self.data.columns:
                    
                    if self.data[col].nunique() == 1:
                        self.constant_cols.append(col)
                    
                    if 'CFLARE' in col or 'Down/Up Ratio' in col or 'ACK Flag Count' in col:
                        self.continuous_cols.append(col)
                    
                    elif self.data[col].nunique() <= 5: 
                        self.discrete_cols.append(col)
                        self.discrete_nums.append(self.data[col].nunique())
                    
                    else:
                        self.continuous_cols.append(col)

                # for c in ['MFLARE','XFLARE']:
                #     self.discrete_cols.remove(c)
                #     self.continuous_cols.append(c)
                self.n_discrete = len(self.discrete_cols)
                self.n_continuous = len(self.continuous_cols)
            else: 
                self.discrete_cols = discrete_cols
                self.continuous_cols = continuous_cols
            
            # print(self.discrete_cols)
            self.data[self.discrete_cols] = self.data[self.discrete_cols].apply(lambda x: pd.factorize(x)[0])

        # ---------- internal 下采样 ----------
        self.internal = 1
        if self.mode == "train":
            # sampling_rate=0.5 → internal=2  表示每隔 2 个样本取 1 个
            self.internal = max(1, int(1 // self.sampling_rate))

    # ---------------- Dataset API ----------------
    def __len__(self):
        base = (len(self.data) - self.win_size) // self.step + 1
        if self.mode == "train":
            return int(base * self.sampling_rate)
        elif self.mode in ("val", "test"):
            return base
        else:                                   # 'pred' / 其他
            return (len(self.data) - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        # ---- 与原逻辑一致：下标 = index * step * internal ----
        if self.mode == "train":
            start = min(index, self.data.shape[0] - self.win_size)
        else:
            start = index * self.step
            start = min(start, self.data.shape[0] - self.win_size)
        end   = start + self.win_size
        # ===== 标签 =====
        if self.mode in ("train", "val", "test"):
            label_arr = self.test_labels.iloc[start:end]
            win_df = self.data.iloc[start:end]
        else:  # 'pred'
            label_arr = self.test_labels.iloc[
                        (start // self.step) * self.win_size: (start // self.step + 1) * self.win_size]
            win_df = self.data.iloc[(start // self.step) * self.win_size: (start // self.step + 1) * self.win_size]

        if self.discrete:
            # ===== 连续特征 =====
            cont_arr = (
                win_df[self.continuous_cols].to_numpy(dtype=np.float32)
                if self.continuous_cols else
                np.empty((self.win_size, 0), dtype=np.float32)
            )

            disc_arr = (
                win_df[self.discrete_cols].to_numpy(dtype=np.float32)
                if self.discrete_cols else
                np.empty((self.win_size, 0), dtype=np.float32)
            )

            return cont_arr, disc_arr, label_arr.to_numpy(dtype=np.float32)
        else:
             # ===== 连续特征 =====
            cont_arr = (
                win_df.to_numpy(dtype=np.float32)
            )

            return cont_arr,  label_arr.to_numpy(dtype=np.float32)



def anomaly_detection_data_provider(
        data, batch_size, win_size=100, step=100,
        mode="train", sampling_rate=1.0, discrete = False,discrete_cols=None,continuous_cols=None):
    """
    与旧版接口保持一致，只把 `step` 透传给 SegLoader
    """
    dataset = SegLoader(data, win_size, 1, mode, sampling_rate, discrete=discrete,discrete_cols=discrete_cols,continuous_cols=continuous_cols)

    shuffle = mode in ("train", "val")
    return DataLoader(dataset,
                      batch_size=batch_size,
                      shuffle=shuffle,
                      num_workers=0,
                      drop_last=False)
