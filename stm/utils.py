import os
from collections import defaultdict
from datetime import datetime, timedelta

import numpy as np
import torch
import torch.nn as nn


class Directory:
    """Directory manager"""

    def __init__(self, dir_main: str) -> None:
        """Initialize

        Parameters
        ----------
        dir_main : str
            Path of main directory (with trailing slash "/")
        """
        if not os.path.exists(dir_main):
            raise FileNotFoundError(f"'{dir_main}' doesn't exist")
        self.main = dir_main
        self.add_dirs(
            code="code/",
            data="data/",
            raw="data/raw/",
            clean="data/clean/",
            config="config/",
            log="log/",
            model="model/",
            result="result/",
        )

    def add_dirs(self, **dir_info: dict) -> None:
        """Add directories

        Parameters
        ----------
        dir_info : dict[str, str]
            Directory name to its relative path (with trailing slash "/")
        """
        for dir_name, dir_relative_path in dir_info.items():
            dir_path = os.path.join(self.main, dir_relative_path)
            setattr(self, dir_name, dir_path)
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)


def get_dates(start_date: str, end_date: str) -> list:
    """Get dates

    Parameters
    ----------
    start_date : str
        Start date in format "%Y%m%d"
    end_date : str
        End date in format "%Y%m%d"

    Returns
    -------
    list[str]
        Dates in format "%Y%m%d"
    """
    dates = []
    s = datetime.strptime(start_date, "%Y%m%d")
    e = datetime.strptime(end_date, "%Y%m%d")
    d = s
    while d <= e:
        dates.append(d.strftime("%Y%m%d"))
        d += timedelta(days=1)
    return dates


def get_times(intervals: list, freq: dict) -> list:
    """Get times

    Parameters
    ----------
    intervals : list[str]
        Time intervals, such as [["093000", "113000"], ["130100", "150000"]]
    freq : dict
        Time frequency, such as {"minutes": 1}

    Returns
    -------
    list[str]
        Times in format "%H%M%S"
    """
    times = []
    for s, e in intervals:
        s, e = datetime.strptime(s, "%H%M%S"), datetime.strptime(e, "%H%M%S")
        t = s
        while t <= e:
            times.append(t.strftime("%H%M%S"))
            t += timedelta(**freq)
    return times


def get_tvt_dates(all_dates: list, partition: list) -> dict:
    """Get training, validation, and testing dates

    Parameters
    ----------
    all_dates : list[str]
        All trading dates in format "%Y%m%d"
    partition : list[str]
        Partition dates (all in format "%Y%m%d"):
            d1: training set start date
            d2: validation set start date
            d3: testing set start date
            d4: testing set end date

    Returns
    -------
    dict[str, list[str]]
        Mode to dates in format "%Y%m%d"
    """
    d1, d2, d3, d4 = map(lambda x: datetime.strptime(x, "%Y%m%d"), partition)
    tvt_dates = defaultdict(list)
    for date in all_dates:
        d = datetime.strptime(date, "%Y%m%d")
        if d < d1:
            continue
        elif d < d2:
            tvt_dates["train"].append(datetime.strftime(d, "%Y%m%d"))
        elif d < d3:
            tvt_dates["valid"].append(datetime.strftime(d, "%Y%m%d"))
        elif d <= d4:
            tvt_dates["test"].append(datetime.strftime(d, "%Y%m%d"))
        else:
            break
    return tvt_dates


def np_ffill(a: np.ndarray, axis: int = 0) -> np.ndarray:
    """Forward fill missing values in an array

    Parameters
    ----------
    a : np.ndarray
        Input array
    axis : int, optional
        Axis along which the missing values are filled, by default 0

    Returns
    -------
    np.ndarray
        Array with missing values filled
    """
    ndim = len(a.shape)
    idx_shape = tuple([slice(None)] + [np.newaxis] * (ndim - axis - 1))
    idx = np.where(~np.isnan(a), np.arange(a.shape[axis])[idx_shape], 0)
    np.maximum.accumulate(idx, axis, out=idx)
    advanced_idx = [
        np.arange(k)[tuple(slice(None) if d == i else np.newaxis for d in range(ndim))]
        for i, k in enumerate(a.shape)
    ]
    advanced_idx[axis] = idx
    return a[tuple(advanced_idx)]


def get_active_fn(active_fn: str, **kwargs: dict) -> nn.Module:
    """Get active function

    Parameters
    ----------
    active_fn : str
        Name of active function
    kwargs : dict
        Keyword arguments for active function

    Returns
    -------
    nn.Module
        Callable module of active function
    """
    active_fn = active_fn.lower()
    if active_fn == "relu":
        return nn.ReLU()
    if active_fn == "sigmoid":
        return nn.Sigmoid()
    if active_fn == "tanh":
        return nn.Tanh()
    if active_fn == "leakyrelu":
        return nn.LeakyReLU(**kwargs)
    if active_fn == "elu":
        return nn.ELU(**kwargs)
    if active_fn == "gelu":
        return nn.GELU(**kwargs)
    if active_fn == "silu":
        return nn.SiLU(**kwargs)
    return ValueError(f"{active_fn} is not supported")


def check_valid_batch(ignore_index: torch.Tensor, n_thresh: int) -> torch.Tensor:
    """Check if there are enough valid samples in each batch

    Parameters
    ----------
    ignore_index : torch.Tensor
        Whether a label should be ignored, B*N
    n_thresh : int
        Minimum number of non-NaN samples

    Returns
    -------
    torch.Tensor
        Batch mask, B
    """
    batch_mask = (~ignore_index).sum(1) < n_thresh
    return batch_mask


def mask_metric_io(
    *ts: tuple,
    mask: torch.Tensor = None,
    n_thresh: int = None,
) -> tuple:
    """Mask inputs and outputs of metric functions

    Parameters
    ----------
    ts: tuple[torch.Tensor, ...]
        Input tensors, B(*N)
    mask : torch.Tensor | None, optional
       Whether a value should be masked, B*N, by default None
    n_thresh : int, optional
        Minimum number of non-NaN samples, by default None

    Returns
    -------
    torch.Tensor | tuple[torch.Tensor, ...]
        Masked tensors, B*N
    """
    if mask is not None:
        if n_thresh is None:  # mask inputs
            ts = tuple(t.masked_fill(mask, float("nan")) for t in ts)
        else:  # mask outputs
            batch_mask = check_valid_batch(mask, n_thresh)
            ts = tuple(t.masked_fill(batch_mask, float("nan")) for t in ts)
    return ts[0] if len(ts) == 1 else ts


def torch_nanstd(
    t: torch.Tensor, dim: list, keepdim: bool = False
) -> torch.Tensor:
    """Calculate standard deviation without considering NaN of a tensor

    Parameters
    ----------
    t : torch.Tensor
        Input tensor
    dim : int | list[int]
        Dimension or dimensions to reduce
    keepdim : bool, optional
        whether the output tensor has `dim` retained or not, by default False

    Returns
    -------
    torch.Tensor
        Standard deviations
    """
    nan_mask = torch.isnan(t)
    n_valid = (~nan_mask).sum(dim, keepdim)
    m = torch.nanmean(t, dim, True)
    sq = torch.where(nan_mask, torch.zeros_like(t), (t - m) ** 2)
    s = torch.sqrt(torch.sum(sq, dim, keepdim) / n_valid)
    s[n_valid == 1] = float("nan")
    return s


def torch_nanzscore(
    t: torch.Tensor, dim: list, eps: float = 1e-6
) -> torch.Tensor:
    """Calculate z-score values

    Parameters
    ----------
    t : torch.Tensor
        Input tensor
    dim : int | list[int]
        Dimension or dimensions to reduce
    eps : float, optional
        Stability constant, by default 1e-6

    Returns
    -------
    torch.Tensor
        Standardized tensor
    """
    m = torch.nanmean(t, dim, True)
    s = torch_nanstd(t, dim, True)
    return (t - m) / (s + eps)


def torch_bpcorr(t1: torch.Tensor, t2: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Batch Pearson correlation

    Parameters
    ----------
    t1 : torch.Tensor
        Input tensor 1, B*N
    t2 : torch.Tensor
        Input tensor 2, B*N
    eps : float, optional
        Stability constant, by default 1e-6

    Returns
    -------
    torch.Tensor
        Pearson correlation, B
    """
    t1 = torch_nanzscore(t1, 1, eps)
    t2 = torch_nanzscore(t2, 1, eps)
    corr = (t1 * t2).nanmean(1)
    return corr


def torch_bscorr(t1: torch.Tensor, t2: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Batch Spearman correlation

    Parameters
    ----------
    t1 : torch.Tensor
        Input tensor 1, B*N
    t2 : torch.Tensor
        Input tensor 2, B*N
    eps : float, optional
        Stability constant, by default 1e-6

    Returns
    -------
    torch.Tensor
        Spearman correlation, B
    """
    nan_mask = torch.isnan(t1) | torch.isnan(t2)
    t1 = t1.masked_fill(nan_mask, float("inf")).nan_to_num()
    t2 = t2.masked_fill(nan_mask, float("inf")).nan_to_num()
    t1_rank = t1.argsort().argsort().to(torch.float32)
    t2_rank = t2.argsort().argsort().to(torch.float32)
    t1_rank.masked_fill_(nan_mask, float("nan"))
    t2_rank.masked_fill_(nan_mask, float("nan"))
    corr = torch_bpcorr(t1_rank, t2_rank, eps)
    return corr


def torch_bcond(
    t: torch.Tensor,
    l: torch.Tensor,
    r: torch.Tensor,
    close_l: bool = False,
    close_r: bool = False,
) -> torch.Tensor:
    """Batch condition between a interval

    Parameters
    ----------
    t : torch.Tensor
        Input tensor, B*N
    l : torch.Tensor
        Left bounds, B
    r : torch.Tensor
        Right bounds, B
    close_l : bool, optional
        Close left bound (l <= t), by default False
    close_r : bool, optional
        Close right bound (t <= r), by default False

    Returns
    -------
    torch.Tensor
        Condition bools, B*N
    """
    cond_l = (t >= l.unsqueeze(1)).bool() if close_l else (t > l.unsqueeze(1)).bool()
    cond_r = (t <= r.unsqueeze(1)).bool() if close_r else (t < r.unsqueeze(1)).bool()
    return cond_l & cond_r


def torch_attn_mask(seq_len: int, prefix_len: int = 0) -> torch.Tensor:
    """Generate attention mask

    Parameters
    ----------
    seq_len : int
        Sequence length
    prefix_len : int, optional
        Prefix length, by default 0

    Returns
    -------
    torch.Tensor
        Mask(i, j) means if i cannot see j
    """
    mask = torch.triu(torch.ones((seq_len, seq_len), dtype=torch.bool), diagonal=1)
    mask[:prefix_len, :prefix_len] = False  # prefix is fully-visible
    return mask
