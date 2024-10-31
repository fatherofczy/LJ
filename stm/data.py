import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn
import yaml
# from pathos.multiprocessing import ProcessingPool
from torch.utils.data import Dataset
from tqdm import tqdm
import datetime
from .utils import Directory, get_dates, get_times, np_ffill

warnings.filterwarnings("ignore")


class Preprocessor:
    """Preprocessing data"""

    def __init__(self, dir: Directory, configs_filename: str = "configs.yaml") -> None:
        """Initialize

        Parameters
        ----------
        dir_main : str
            Path of main directory (with trailing slash "/")
        configs_filename : str, optional
            Filename of configurations, by default "configs.yaml"
        """
        # directory
        self.dir = dir
        # configurations
        with open(self.dir.config + configs_filename, "r") as f:
            configs = yaml.safe_load(f)
        self.dir_scan = configs["prep"]["path"]["dir_scan"]
        self.dir_x = configs["prep"]["path"]["dir_x"]
        self.dir_y = configs["prep"]["path"]["dir_y"]
        self.dir_save = configs["prep"]["path"]["dir_save"]
        self.dir.add_dirs(save=self.dir_save)
        self.date_range = configs["prep"]["date_range"]
        self.intervals = configs["prep"]["intervals"]
        self.freq = configs["prep"]["freq"]
        self.n_parts = configs["prep"]["by_parts"]["n_parts"]
        self.curr_part = configs["prep"]["by_parts"]["curr_part"]
        self.processes = configs["prep"]["processes"]
        save_suffix = configs["prep"]["save_suffix"]
        self.save_suffix = f"_{save_suffix}" if len(save_suffix) else ""

    def scan(self, ticker_col: str = "ticker") -> None:
        """Scan data

        1) get all trading dates
        2) get all trading times
        3) get all tickers (union)

        Parameters
        ----------
        ticker_col : str, optional
            Column of tickers, by default "ticker"
        """
        if self.dir_scan is None:
            self.dir_scan = self.dir.raw
        # scan files
        self.dates = []
        self.times = get_times(self.intervals, self.freq)
        self.tickers = set()
        dates = get_dates(*self.date_range)
        for d in tqdm(dates):
            try:
                df = pd.read_feather(self.dir_scan + f"{d[:4]}/{d}/e100000.feather")
                self.dates.append(d)
                self.tickers = self.tickers.union(set(df[ticker_col]))
            except Exception:
                continue
        self.tickers = sorted(list(self.tickers))  # fix order
        # print
        print("number of dates:", len(self.dates))
        print("number of times:", len(self.times))
        print("number of tickers:", len(self.tickers))

    def clean(
        self,
        x_cols: list,
        y_col: str,
        ticker_col_x: str = "ticker",
        ticker_col_y: str = "ticker",
    ) -> None:
        """Clean data

        1) merge tickers to the union of tickers at each time
        2) get features and labels
        3) get untradable signals from limit up and limit down
        4) forward fill NaN of features every day (NaN at fisrt time cannot be filled)
        5) get padding mask for tickers from missing feature values
        6) get ignored index from any missing values in features, labels, and signals
        7) save features (x), labels (y), padding mask (p) ingored index (i) using mmap

        Parameters
        ----------
        x_cols : list
            Columns of features
        y_col : str
            Column of labels
        ticker_col_x : str
            Column of tickers in dataframes in `dir_x`, by default "ticker"
        ticker_col_y : str
            Column of tickers in dataframes in `dir_y`, by default "ticker"
        """
        if self.dir_x is None:
            self.dir_x = self.dir.raw
        if self.dir_y is None:
            self.dir_y = self.dir.raw
        if self.dir_save is None:
            self.dir_save = self.dir.clean
        # all tickers
        df_ticker = pd.DataFrame(self.tickers, columns=["ticker"])

        # helper function for each day data cleaning
        def helper(d: str) -> None:
            """Clean data for one day

            Parameters
            ----------
            d : str
                Date
            """
            x, y, s = [], [], []
            for t in tqdm(self.times):
                # read
                try:
                    # features
                    df_x = pd.read_feather(self.dir_x + f"{d[:4]}/{d}/e{t}.feather")
                    df_x.rename(columns={ticker_col_x: "ticker"}, inplace=True)
                    df_x = df_ticker.merge(df_x, "left", "ticker")
                    xx = df_x[x_cols].values
                    # labels
                    df_y = pd.read_feather(self.dir_y + f"{d[:4]}/{d}/e{t}.feather")
                    df_y.rename(columns={ticker_col_y: "ticker"}, inplace=True)
                    df_y = df_ticker.merge(df_y, "left", "ticker")
                    yy = df_y[y_col].values
                    ss = (df_y["isZT"].values == 1) | (df_y["isDT"].values == 1)
#                     xx = (xx - np.nanmean(xx, axis=0)) / (np.nanstd(xx, axis=0) + 1e-6)  # 截面标准化
#                     yy = yy - np.nanmean(yy, axis=0) 
                except Exception as e:
                    print(e)
                    xx = np.full((len(self.tickers), len(x_cols)), np.nan)
                    yy = np.full(len(self.tickers), np.nan)
                    ss = np.full(len(self.tickers), True)
                # append
                x.append(xx)
                y.append(yy)
                s.append(ss)
            # stack
            x = np.stack(x)  # T*N*C
            y = np.stack(y)  # T*N
            s = np.stack(s)  # T*N
            # ffill
            x = np_ffill(x)
            # check NaN
            p = np.all(np.isnan(x), (0, 2))  # N
            i = p.reshape((1, -1)) | np.isnan(y) | s  # T*N
            # fill NaN
            np.nan_to_num(x, False)
            np.nan_to_num(y, False)
            # convert to float32
            x = x.astype(np.float32, copy=False)
            y = y.astype(np.float32, copy=False)
            # write
            block=64*1024
            with open(self.dir_save + f"x_{d}.dat", "wb") as f:
                for j in range(len(x)):  # since x is too large
                    f.write(x[j].tobytes())
                f.flush()
            with open(self.dir_save + f"y_{d}.dat", "wb") as f:
                f.write(y.tobytes())
                f.flush()
            with open(self.dir_save + f"p_{d}.dat", "wb") as f:
                f.write(p.tobytes())
                f.flush()
            with open(self.dir_save + f"i_{d}.dat", "wb") as f:
                f.write(i.tobytes())
                f.flush()

        # clean data by parts and by multiprocessing
#         idx_l = len(self.dates) * (self.curr_part - 1) // self.n_parts
#         idx_r = len(self.dates) * self.curr_part // self.n_parts
#         helper(self.dates[0])
        with ProcessingPool(nodes=self.processes) as p:
            p.map(helper, self.dates)
        # save data info
        np.savez_compressed(
            self.dir.config + f"data_info{self.save_suffix}.npz",
            path=self.dir_save,
            dates=self.dates,
            times=self.times,
            tickers=self.tickers,
            x_cols=x_cols,
            y_col=y_col,
        )

class StockMmapDataset(Dataset):
    """Dataset based on memory-mapped files"""

    def __init__(
        self,
        data_path: str,
        dates: list,
        times: list,
        N: int,
        L: int,
        C: int,
        stocknum:int,
        start:int,
        end:int,
        x_keep: list = None,
        y_keep: list = None,
        clip: list = None,
        extending: bool = False,
        sequence:bool=False,
        time_range:list=None,
        **args
    ) -> None:

        self.data_path = data_path
        self.dates, self.times = dates, times
        self.N, self.L, self.C = N, L, C
        self.stocknum=stocknum
        self.groupnum=int(np.ceil(self.N/stocknum))
        self.x_keep = list(range(len(times))) if x_keep is None else x_keep
        self.y_keep = list(range(len(times))) if y_keep is None else y_keep
        self.time_range=list(range(len(y_keep))) if time_range is None else time_range
        self.S = len(self.time_range)  # number of valid labels per day
        self.index_base = self.L // self.S + 1  # the first valid date index
        self.clip = clip
        self.extending = extending
        self.sequence=sequence
        
#         self.x=np.memmap('/dfs/data/data/cat/x.dat',mode='r',dtype=np.float32,shape=(427,238,3759,170))[start:end,self.time_range]
#         self.y=np.memmap('/dfs/data/data/cat/y.dat',mode='r',dtype=np.float32,shape=(427,238,3759))[start:end,self.time_range]
#         self.i=np.memmap('/dfs/data/data/cat/i.dat',mode='r',dtype=np.bool_,shape=(427,238,3759))[start:end,self.time_range]
#         self.p=np.memmap('/dfs/data/data/cat/p.dat',mode='r',dtype=np.bool_,shape=(427,3759))
    def __len__(self) -> int:
        # make sure that we start from a day with enough lookback data
        return (len(self.dates) - self.index_base) * (1 if self.extending else self.S)*self.groupnum
    def __getitem__(self, index: int) -> tuple:
        # extending window
       
        # get index for date and time
        time,group=index//self.groupnum,index%self.groupnum
        j, k = time // self.S + self.index_base, time % self.S
        x=np.memmap('/dfs/data/data/cat/x.dat',mode='r',dtype=np.float32,shape=(427,238,3759,170))[start:end,self.time_range][j-self.index_base:j+1]
        x=x.reshape(-1,*x.shape[2:])
        stockslice=range(min(group*self.stocknum,self.N-self.stocknum),min(self.N,(1+group)*self.stocknum))
        if k != self.S - 1:
            x = x[: -(self.S - k - 1), :, :]  # since x only has one more 240 than y
        # keep lookback times
        x = x[-self.L :,stockslice,:]  # L*N*C
        # transpose features
        x = np.transpose(x, (1, 0, 2))  # N*L*C
        # read labels
        if self.sequence:
            y=np.memmap('/dfs/data/data/cat/y.dat',mode='r',dtype=np.float32,shape=(427,238,3759))[start:end,self.time_range][j-self.index_base:j+1]
            y=y.reshape(-1,*y.shape[2:])
            if k!=self.S-1:
                y = y[: -(self.S - k - 1), :]
#                 y=y[max(-(k+1),-self.L):,stockslice]
            y=y[-self.L:,stockslice]
            y=y.transpose(1,0)
        # keep lookback times
        else:
            y=np.memmap('/dfs/data/data/cat/y.dat',mode='r',dtype=np.float32,shape=(427,238,3759))[start:end,self.time_range][j,k,stockslice]
        # read padding mask
        pad_mask = np.full(self.N, False)
        for d in range(j-self.index_base,j+1):
            pad_mask|=np.memmap('/dfs/data/data/cat/p.dat',mode='r',dtype=np.bool_,shape=(427,3759))[d]
        ignore_index=np.memmap('/dfs/data/data/cat/i.dat',mode='r',dtype=np.bool_,shape=(427,238,3759))[start:end,self.time_range][j]
        x = torch.from_numpy(x).float()
        y = torch.from_numpy(y).float()
        pad_mask = torch.from_numpy(pad_mask).bool()
        ignore_index = torch.from_numpy(ignore_index).bool()
        pad_mask=pad_mask[stockslice]
        ignore_index=ignore_index[k,stockslice]
        # clipS
        if self.clip is not None:
            x = x.clamp(*self.clip)
        date=self.dates[j]
        time=self.times[k]
        weekday=datetime.date(int(date[:4]),int(date[4:6]),int(date[6:])).weekday()
        time_info=torch.tensor((int(date[4:6]),int(date[6:]),weekday,int(time[:2]),int(time[2:4])))
        time_info=time_info.unsqueeze(0).expand(self.N,5)
        return x, y, pad_mask, ignore_index,time_info

            
class MmapDataset(Dataset):
    """Dataset based on memory-mapped files"""

    def __init__(
        self,
        data_path: str,
        dates: list,
        times: list,
        N: int,
        L: int,
        C: int,
        start:int,
        end:int,
        stocknum: int,
        x_keep: list = None,
        y_keep: list = None,
        clip: list = None,
        extending: bool = False,
        sequence:bool=False,
        time_range:list=None,
        **args
    ) -> None:
        """Initialize

        Parameters
        ----------
        dates : list
            Trading dates for this dataset
        times : list
            Trading times for each day
        N : int
            Size of tickers
        L : int
            Size of lookbacks
        C : int
            Size of characteristics
        x_keep : list, optional
            Index of `times` allowed to be used in features, by default None
        y_keep : list, optional
            Index of `times` allowed to be used in labels, by default None
        clip : list[float] | None, optional
            Clip features to range `clip`
        extending : bool, optional
            Set to `True` to extend the lookback until the end of everyday, else the
            lookback is rolling, by default False
        """
        self.data_path = data_path
        self.dates, self.times = dates, times
        self.N, self.L, self.C = N, L, C
        self.x_keep = list(range(len(times))) if x_keep is None else x_keep
        self.y_keep = list(range(len(times))) if y_keep is None else y_keep
        self.time_range=self.y_keep if time_range is None else time_range
        self.S = len(self.time_range)   # number of valid labels per day
        self.index_base = self.L // self.S + 1  # the first valid date index
        self.clip = clip
        self.extending = extending

    def __len__(self) -> int:
        """Get length

        Returns
        -------
        int
            Length of dataset
        """
        # make sure that we start from a day with enough lookback data

        return (len(self.dates) - self.index_base) * (1 if self.extending else self.S)

    def __getitem__(self, index: int) -> tuple:
        """Get item

        Parameters
        ----------
        index : int
            Index of the item

        Returns
        -------
        tuple[torch.Tensor, ...]
            x_enc : torch.Tensor
                Features for encoder, N*L*C
            x_dec : torch.Tensor
                Features for decoder, N*S*C
            x : torch.Tensor
                Features, N*L*C
            y_dec : torch.Tensor
                Labels for decoder, N*S
            y : torch.Tensor
                Labels, N
            enc_pad_mask : torch.Tensor
                Padding mask for `x_enc`, N
            dec_pad_mask : torch.Tensor
                Padding mask for `x_dec`, N
            pad_mask : torch.Tensor
                Padding mask for `x`, N
            ignore_index : torch.Tensor (`extending=True`)
                Whether a label should be ignored, N*S
            ignore_index : torch.Tensor (`extending=False`)
                Whether a label should be ignored, N
        """
        # extending window
        if self.extending:
            # get index for date
            j = index + self.index_base
            # read features for encoder
            x_enc = [
                np.memmap(
                    f"{self.data_path}x_{d}.dat",
                    dtype=np.float32,
                    mode="r",
                    shape=(len(self.times), self.N, self.C),
                )
                for d in self.dates[j - self.index_base : j]
            ]
            # keep valid times
            x_enc = np.concatenate([i[self.x_keep] for i in x_enc])  # (D*T)*N*C
            # keep lookback times
            x_enc = x_enc[-self.L :]
            # transpose features
            x_enc = np.transpose(x_enc, (1, 0, 2))  # N*L*C
            # read features for decoder
            x_dec = np.memmap(
                f"{self.data_path}x_{self.dates[j]}.dat",
                dtype=np.float32,
                mode="r",
                shape=(len(self.times), self.N, self.C),
            )
            # keep valid times
            x_dec = x_dec[self.y_keep]  # S*N*C
            # transpose features
            x_dec = np.transpose(x_dec, (1, 0, 2))  # N*S*C
            # read labels
            y_dec = np.memmap(
                f"{self.data_path}y_{self.dates[j]}.dat",
                dtype=np.float32,
                mode="r",
                shape=(len(self.times), self.N),
            )
            # get corresponding labels
            y_dec = y_dec[self.y_keep, :]  # S*N
            # transpose labels
            y_dec = np.transpose(y_dec, (1, 0))  # N*S
            # read padding mask for `x_enc`
            enc_pad_mask = np.full(self.N, False)  # N
            for d in self.dates[j - self.index_base : j]:
                enc_pad_mask |= np.memmap(
                    f"{self.data_path}p_{d}.dat",
                    dtype=bool,
                    mode="r",
                    shape=self.N,
                )
            # read padding mask for `x_dec`
            dec_pad_mask = np.memmap(
                f"{self.data_path}p_{self.dates[j]}.dat",
                dtype=bool,
                mode="r",
                shape=self.N,
            )  # N
            # read ignored index
            ignore_index = np.memmap(
                f"{self.data_path}i_{self.dates[j]}.dat",
                dtype=bool,
                mode="r",
                shape=(len(self.times), self.N),
            )
            # get corresponding ignored index
            ignore_index = ignore_index[self.y_keep, :]  # S*N
            # transpose ignored index
            ignore_index = np.transpose(ignore_index, (1, 0))  # N*S
            # to tensor
            x_enc = torch.from_numpy(x_enc).float()
            x_dec = torch.from_numpy(x_dec).float()
            y_dec = torch.from_numpy(y_dec).float()
            enc_pad_mask = torch.from_numpy(enc_pad_mask).bool()
            dec_pad_mask = torch.from_numpy(dec_pad_mask).bool()
            ignore_index = torch.from_numpy(ignore_index).bool()
            # clip
            if self.clip is not None:
                x_enc = x_enc.clamp(*self.clip)
                x_dec = x_dec.clamp(*self.clip)
            return x_enc, x_dec, y_dec, enc_pad_mask, dec_pad_mask, ignore_index
        # rolling window
        else:
            # get index for date and time

            j, k = index // self.S + self.index_base, index % self.S
            # read features
            x = [
                np.memmap(
                    f"{self.data_path}x_{d}.dat",
                    dtype=np.float32,
                    mode="r",
                    shape=(len(self.times), self.N, self.C),
                )
                for d in self.dates[j - self.index_base : j + 1]
            ]
            # keep valid times
            x = np.concatenate([i[self.time_range] for i in x])  # (D*T)*N*C
            # drop future times
            if k != self.S - 1:
                x = x[: -(self.S - k - 1), :, :]  # since x only has one more 240 than y
            # keep lookback times
            x = x[-self.L :]  # L*N*C
            # transpose features
            x = np.transpose(x, (1, 0, 2))  # N*L*C
            # read labels
            y = np.memmap(
                f"{self.data_path}y_{self.dates[j]}.dat",
                dtype=np.float32,
                mode="r",
                shape=(len(self.times), self.N),
            )
            # get corresponding labels
            y = y[self.time_range[k], :]  # N
            # read padding mask
            pad_mask = np.full(self.N, False)  # N
            for d in self.dates[j - self.index_base : j + 1]:
                pad_mask |= np.memmap(
                    f"{self.data_path}p_{d}.dat",
                    dtype=bool,
                    mode="r",
                    shape=self.N,
                )
            # read ignored index
            ignore_index = np.memmap(
                f"{self.data_path}i_{self.dates[j]}.dat",
                dtype=bool,
                mode="r",
                shape=(len(self.times), self.N),
            )
            # get corresponding ignored index
            ignore_index = ignore_index[self.time_range[k], :]  # N
            # to tensor
            x = torch.from_numpy(x).float()
            y = torch.from_numpy(y).float()
            pad_mask = torch.from_numpy(pad_mask).bool()
            ignore_index = torch.from_numpy(ignore_index).bool()
            # clip
            if self.clip is not None:
                x = x.clamp(*self.clip)
            date=self.dates[j]
            time=self.times[k]
            weekday=datetime.date(int(date[:4]),int(date[4:6]),int(date[6:])).weekday()
            time_info=torch.tensor((int(date[4:6]),int(date[6:]),weekday,int(time[:2]),int(time[2:4])))
            time_info=time_info.unsqueeze(0).expand(self.N,5)
            return x, y, pad_mask, ignore_index,time_info

class IntraMmapDataset(Dataset):
    """Dataset based on memory-mapped files"""

    def __init__(
        self,
        data_path: str,
        dates: list,
        times: list,
        N: int,
        L: int,
        C: int,
        start:int,
        end:int,
        stocknum: int,
        x_keep: list = None,
        y_keep: list = None,
        clip: list = None,
        extending: bool = False,
        sequence:bool=False,
        time_range:list=None,
        intraday=False,
        **args
    ) -> None:
        self.data_path = data_path
        self.dates, self.times = dates, times
        self.N, self.L, self.C = N, L, C
        self.x_keep = list(range(len(times))) if x_keep is None else x_keep
        self.y_keep = list(range(len(times))) if y_keep is None else y_keep
        self.time_range=self.y_keep if time_range is None else time_range
        self.S = len(self.time_range)   # number of valid labels per day
        self.clip_S=self.S-self.L+1
 
        assert self.L<=self.S
#        the first valid date index
        self.clip = clip
        self.extending = extending

    def __len__(self) -> int:

        return (len(self.dates)) * (1 if self.extending else self.clip_S)
            

    def __getitem__(self, index: int) -> tuple:

        # get index for date and time

        j, k = index // self.clip_S, index % self.clip_S
        x=np.memmap(
                f"{self.data_path}x_{self.dates[j]}.dat",
                dtype=np.float32,
                mode="r",
                shape=(len(self.times), self.N, self.C),
            )[self.time_range]
        # drop future times
        if k != self.clip_S - 1:
            x = x[: -(self.clip_S - k - 1), :, :]  # since x only has one more 240 than y
        # keep lookback times
        x = x[-self.L :]  # L*N*C
        # transpose features
        x = np.transpose(x, (1, 0, 2))  # N*L*C
        # read labels
        y = np.memmap(
            f"{self.data_path}y_{self.dates[j]}.dat",
            dtype=np.float32,
            mode="r",
            shape=(len(self.times), self.N),
        )
        # get corresponding labels
        y = y[self.time_range[k]+self.L-1, :]  # N
        # read padding mask
        pad_mask = np.full(self.N, False)  # N
        pad_mask|= np.memmap(
                f"{self.data_path}p_{self.dates[j]}.dat",
                dtype=bool,
                mode="r",
                shape=self.N,
            )
        # read ignored index
        ignore_index = np.memmap(
            f"{self.data_path}i_{self.dates[j]}.dat",
            dtype=bool,
            mode="r",
            shape=(len(self.times), self.N),
        )
        # get corresponding ignored index
        ignore_index = ignore_index[self.time_range[k]+self.L-1, :]  # N
        # to tensor
        x = torch.from_numpy(x).float()
        y = torch.from_numpy(y).float()
        pad_mask = torch.from_numpy(pad_mask).bool()
        ignore_index = torch.from_numpy(ignore_index).bool()
        # clip
        if self.clip is not None:
            x = x.clamp(*self.clip)
        date=self.dates[j]
        time=self.times[k]
        weekday=datetime.date(int(date[:4]),int(date[4:6]),int(date[6:])).weekday()
        time_info=torch.tensor((int(date[4:6]),int(date[6:]),weekday,int(time[:2]),int(time[2:4])))
        time_info=time_info.unsqueeze(0).expand(self.N,5)
        return x, y, pad_mask, ignore_index,time_info

class SequenceMmapDataset(Dataset):
    """Dataset based on memory-mapped files"""

    def __init__(
        self,
        data_path: str,
        dates: list,
        times: list,
        N: int,
        L: int,
        C: int,
        stocknum:int,
        start:int,
        end:int,
        day:int=64,
        x_keep: list = None,
        y_keep: list = None,
        clip: list = None,
        extending: bool = False,
        sequence:bool=False,

        time_range:list=None,
        
    ) -> None:

        self.data_path = data_path
        self.dates, self.times = dates, times
        self.N, self.L, self.C = N, L, C
        self.stocknum=stocknum
        self.groupnum=int(np.ceil(self.N/stocknum))
        self.x_keep = list(range(len(times))) if x_keep is None else x_keep
        self.y_keep = list(range(len(times))) if y_keep is None else y_keep
        self.time_range=list(range(len(y_keep))) if time_range is None else time_range
        self.S = len(self.time_range)  # number of valid labels per day
        self.index_base = self.L // self.S + 1  # the first valid date index
        self.clip = clip
        self.extending = extending
        self.sequence=sequence
        
#         self.x=np.memmap('/dfs/data/data/cat/x.dat',mode='r',dtype=np.float32,shape=(427,238,3759,170))[start:end,self.time_range]
#         self.y=np.memmap('/dfs/data/data/cat/y.dat',mode='r',dtype=np.float32,shape=(427,238,3759))[start:end,self.time_range]
#         self.i=np.memmap('/dfs/data/data/cat/i.dat',mode='r',dtype=np.bool_,shape=(427,238,3759))[start:end,self.time_range]
#         self.p=np.memmap('/dfs/data/data/cat/p.dat',mode='r',dtype=np.bool_,shape=(427,3759))
    def __len__(self) -> int:
        # make sure that we start from a day with enough lookback data
        return (len(self.dates) - self.index_base) * (1 if self.extending else self.S)*self.groupnum
    def __getitem__(self, index: int) -> tuple:
        # extending window
       
        # get index for date and time
        time,group=index//self.groupnum,index%self.groupnum
        j, k = time // self.S + self.index_base, time % self.S
        x=np.memmap('/dfs/data/data/cat/x.dat',mode='r',dtype=np.float32,shape=(427,238,3759,170))[start:end,self.time_range][j-self.index_base:j+1]
        x=x.reshape(-1,*x.shape[2:])
        stockslice=range(min(group*self.stocknum,self.N-self.stocknum),min(self.N,(1+group)*self.stocknum))
        if k != self.S - 1:
            x = x[: -(self.S - k - 1), :, :]  # since x only has one more 240 than y
        # keep lookback times
        x = x[-self.L :,stockslice,:]  # L*N*C
        # transpose features
        x = np.transpose(x, (1, 0, 2))  # N*L*C
        # read labels
        if self.sequence:
            y=np.memmap('/dfs/data/data/cat/y.dat',mode='r',dtype=np.float32,shape=(427,238,3759))[start:end,self.time_range][j-self.index_base:j+1]
            y=y.reshape(-1,*y.shape[2:])
            if k!=self.S-1:
                y = y[: -(self.S - k - 1), :]
#                 y=y[max(-(k+1),-self.L):,stockslice]
            y=y[-self.L:,stockslice]
            y=y.transpose(1,0)
        # keep lookback times
        else:
            y=np.memmap('/dfs/data/data/cat/y.dat',mode='r',dtype=np.float32,shape=(427,238,3759))[start:end,self.time_range][j,k,stockslice]
        # read padding mask
        pad_mask = np.full(self.N, False)
        for d in range(j-self.index_base,j+1):
            pad_mask|=np.memmap('/dfs/data/data/cat/p.dat',mode='r',dtype=np.bool_,shape=(427,3759))[d]
        ignore_index=np.memmap('/dfs/data/data/cat/i.dat',mode='r',dtype=np.bool_,shape=(427,238,3759))[start:end,self.time_range][j]
        x = torch.from_numpy(x).float()
        y = torch.from_numpy(y).float()
        pad_mask = torch.from_numpy(pad_mask).bool()
        ignore_index = torch.from_numpy(ignore_index).bool()
        pad_mask=pad_mask[stockslice]
        ignore_index=ignore_index[k,stockslice]
        # clipS
        if self.clip is not None:
            x = x.clamp(*self.clip)
        date=self.dates[j]
        time=self.times[k]
        weekday=datetime.date(int(date[:4]),int(date[4:6]),int(date[6:])).weekday()
        time_info=torch.tensor((int(date[4:6]),int(date[6:]),weekday,int(time[:2]),int(time[2:4])))
        time_info=time_info.unsqueeze(0).expand(self.N,5)
        return x, y, pad_mask, ignore_index,time_info
