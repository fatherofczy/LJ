import sys
sys.path.append('/dfs/data/package')
import inspect
import numpy as np
import os
import random
from abc import ABC, abstractmethod
from torch.profiler import profile, record_function, ProfilerActivity
import torchvision.models as models
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import yaml
from torch.nn.parallel import DistributedDataParallel
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
from torch.optim.lr_scheduler import _LRScheduler as LRScheduler
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
import copy
from .backtesting import Backtester
from .data import MmapDataset, StockMmapDataset, IntraMmapDataset
from ..utils import Directory, get_tvt_dates
from tensorboardX import SummaryWriter
from .VNet.VNet import VNet
from prefetch_generator import BackgroundGenerator
torch.autograd.set_detect_anomaly(True)

class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())

def shoot_infs(inp_tensor):
    """Replaces infs by maximum of tensor"""
    mask_inf = torch.isinf(inp_tensor)
    ind_inf = torch.nonzero(mask_inf, as_tuple=False)
    if len(ind_inf) > 0:
        for ind in ind_inf:
            if len(ind) == 2:
                inp_tensor[ind[0], ind[1]] = 0
            elif len(ind) == 1:
                m = torch.max(inp_tensor)
                for ind in ind_inf:
                    if len(ind) == 2:
                        inp_tensor[ind[0], ind[1]] = m
                    elif len(ind) == 1:
                        inp_tensor[ind[0]] = m
    return inp_tensor
def sinkhorn(Q, n_iters=3, epsilon=0.01):
    # epsilon should be adjusted according to logits value's scale
    with torch.no_grad():
        Q = shoot_infs(Q)
        Q = torch.exp(Q / epsilon)
        for i in range(n_iters):
            Q /= Q.sum(dim=0, keepdim=True)
            Q /= Q.sum(dim=1, keepdim=True)
    return Q

class Pipeline(ABC):
    """Pipeline for deep learning on return prediction"""
    
    def __init__(self, 
                 dir: Directory, 
                 configs_filename: str = "configs.yaml", 
                 verbose: bool = False) -> None:
        """Initialize
        
        Parameters
        ----------
        dir : Directory
            Directory manager
        configs_filename : str, optional
            Filename of configurations, by default "configs.yaml"
        verbose : bool, optional
            Verbosity mode, by default False
        """
        # read files
        with open(dir.config + configs_filename) as f:
            configs = yaml.safe_load(f)
        save_suffix = configs["prep"]["save_suffix"]
        save_suffix = f"_{save_suffix}" if len(save_suffix) else ""
        data_info_filename = f"data_info{save_suffix}.npz"
        data_info = np.load(dir.config + data_info_filename, allow_pickle=True)
        
        # data
        self.data_path = str(data_info["path"])
        self.dates = data_info["dates"]
        self.times = data_info["times"]
        self.tickers = data_info["tickers"]
        # more initialization code may be here
        self.partition = configs["data"]["partition"]
        self.x_keep = [i for i in range(241) if i not in configs["data"]["x_skip"]]
        self.y_keep = [i for i in range(241) if i not in configs["data"]["y_skip"]]
        self.clip = configs["data"]["clip"]
        self.writer = SummaryWriter(logdir=configs["model"]["writer"])
        self.extending = configs["data"]["extending"]
        self.stocknum = configs["data"]["stocknum"]
        self.sequence = configs["data"]["sequence"]

        # seed
        self.S = len(self.y_keep)
        self.input_shape = configs["model"]["input_shape"]
        self.index_base = self.input_shape["L"] // self.S + 1
        self.tvt_dates = get_tvt_dates(self.dates, self.partition)

        # optim
        self.seed = configs["optim"]["seed"]
        self.train_batch_size = configs["optim"]["train_batch_size"]
        self.valid_batch_size = configs["optim"]["valid_batch_size"]
        self.num_workers = configs["optim"]["num_workers"]
        self.ddp = configs["optim"]["ddp"]
        self.lr = configs["optim"]["lr"]
        self.betas = configs["optim"]["betas"]
        self.weight_decay = configs["optim"]["weight_decay"]
        self.fused = configs["optim"]["fused"]
        self.epochs = configs["optim"]["epochs"]
        self.sub_epochs = configs["optim"]["sub_epochs"]
        self.warmup_end = configs["optim"]["warmup_end"]
        self.warmup_base = configs["optim"]["warmup_base"]
        self.anneal_start = configs["optim"]["anneal_start"]
        self.anneal_eta_min = configs["optim"]["anneal_eta_min"]
        self.enable_valid = configs["optim"]["enable_valid"]

        # model
        t = configs["model"]["type"]
        n = configs["model"]["name"]
        v = configs["model"]["version"]
        self.model_name = f"{t}_{n}_{v}"

        self.base_params = configs["model"]["base_params"]

        # backtest
        self.n_thresh = configs["backtest"]["n_thresh"]
        self.bt_eps = configs["backtest"]["eps"]
        self.select_1 = configs["backtest"]["select_1"]
        self.select_2 = configs["backtest"]["select_2"]
        self.pct_s = configs["backtest"]["pct_s"]
        self.n_groups = configs["backtest"]["n_groups"]
        self.scale = configs["backtest"]["scale"]
        self.metrics = configs["backtest"]["metrics"]
        self.rho = 0.99
        self.lambda2 = 2
        self.scale = configs["backtest"]["scale"]
        self.metrics = configs["backtest"]["metrics"]
        self.rho = 0.99
        self.lambda2 = 2

        # general
        dir_m_exp = dir.model + self.model_name + "/"
        dir_r_exp = dir.result + self.model_name + "/"
        dir.add_dirs(m=dir_m_exp, r_exp=dir_r_exp)
        self.dir = dir
        self.verbose = verbose
        self.N = configs["model"]["input_shape"]["N"]

        self.num_states = 5
        self._set_seed()
        # device
        self._set_device()

        # scheme
        self.model_keys = ["class", "pos", "neg"]
        self.scheme = configs["scheme"]["type"]
        self.meta = configs["scheme"]["meta"]
        self.loss_func = getattr(self, self.scheme + "_calc_loss")
        self.eval_func = getattr(self, self.scheme + "_prep_eval")

        def _set_seed(self) -> None:
            """Set random seed"""
            os.environ["PYTHONHASHSEED"] = str(self.seed)
            random.seed(self.seed)
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
            torch.cuda.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True

        def _set_device(self) -> None:
            """Set device"""
            if self.ddp:  # DDP
                dist.init_process_group(backend="nccl")
                self.world_size = dist.get_world_size()
                self.local_rank = dist.get_rank() % torch.cuda.device_count()
                torch.cuda.set_device(self.local_rank)
                self.verbose = self.verbose and self.local_rank == 0
            else:  # DP
                self.local_rank = None
                self.use_cuda = torch.cuda.is_available()
                if self.use_cuda:
                    self.device = torch.device("cuda")
                else:
                    self.device = torch.device("cpu")
        self.local_rank = None
        self.use_cuda = torch.cuda.is_available()
        if self.use_cuda:  # cuda
            self.device = torch.device("cuda")
            if self.verbose:
                print("number of GPUs:", torch.cuda.device_count())
                print("GPU name:", torch.cuda.get_device_name(0))
                print("GPU is on")
        else:  # cpu
            self.device = torch.device("cpu")
            if self.verbose:
                print("GPU is off, using CPU instead")

        def _to_device(self, model) -> None:
            """Model to device"""
            if self.ddp:  # DDP
                model = model.to(self.local_rank)
                model = DistributedDataParallel(
                    module=model,
                    device_ids=[self.local_rank],
                    output_device=self.local_rank,
                    find_unused_parameters=False,
                )
            else:  # DP
                model.to(self.device)
                # model = nn.DataParallel(model)
            return model

        def _gen_dataloader(self, dataset: Dataset, mode: str) -> DataLoader:
            """Generate dataloader

            Parameters
            ----------
            dataset : Dataset
                Dataset from which to load the data
            mode : str
                Operation mode, choose from ["train", "valid", "test"]

            Returns
            -------
            DataLoader
                Dataloader
            """
            if mode not in ["train", "valid", "test"]:
                raise ValueError(f"{mode} is not supported")

            if self.ddp:
                shuffle = False
                sampler = DistributedSampler(dataset, shuffle=(mode == "train"))
            else:
                shuffle = True if mode == "train" else False
                sampler = None

            dataloader = DataLoaderX(
                dataset=dataset,
                batch_size=self.train_batch_size if mode == "train" else self.valid_batch_size,
                shuffle=shuffle,
                sampler=sampler,
                num_workers=self.num_workers,
                pin_memory=True,
                collate_fn=self.collate_fn if isinstance(dataset, StockMmapDataset) else None,
                drop_last=True if mode == "train" else False,
            )

            # Code to set up DataLoader based on the mode and other settings

            return dataloader
        def _dist_gather(self, t: torch.Tensor, mode: str) -> torch.Tensor:
            """Gather tensors distributed in different GPUs

            Parameters
            ----------
            t : torch.Tensor
                Local tensor, batch first
            mode : str
                Operation mode, choose from ["train", "valid", "test"]

            Returns
            -------
            torch.Tensor
                Tensors, batch first
            """
            # store all tensors
            all_t = [torch.empty_like(t) for _ in range(self.world_size)]
            # gather tensors to rank 0
            dist.gather(t, all_t if self.local_rank == 0 else None, 0)
            # process on rank 0
            if self.local_rank == 0:
                # adjust to correct sample order
                order = shape = (1, 0, *range(2, t.ndim + 1), -1, *t.shape[1:])
                return torch.cat(all_t, dim=0).permute(order)
            else:
                return None
        def _init_dataloaders(self, dataset_type: type) -> None:
            """Initialize dataloaders
            
            Parameters
            ----------
            dataset_type : type[Dataset]
                Dataset type
            """

            # generate datasets
            mode_choice = ['train', 'valid', 'test']
            start_index = [0, len(self.tvt_dates['train']), len(self.tvt_dates['train']) + len(self.tvt_dates['valid'])]
            end_index = [len(self.tvt_dates['train']), len(self.tvt_dates['train']) + len(self.tvt_dates['valid']), len(self.tvt_dates['train']) + len(self.tvt_dates['valid']) + len(self.tvt_dates['test'])]
            start_index = dict(zip(mode_choice, start_index))
            end_index = dict(zip(mode_choice, end_index))
            
            datasets = {
                mode: dataset_type(
                    data_path=self.data_path,
                    dates=dates,
                    times=self.times,
                    x_keep=self.x_keep,
                    y_keep=self.y_keep,
                    **self.input_shape,
                    start=start_index[mode],
                    end=end_index[mode],
                    stocknum=self.stocknum,
                    clip=self.clip,
                    extending=self.extending,
                    sequence=self.sequence
                ) for mode, dates in self.tvt_dates.items()
            }

            self.tvt_sizes = {mode: len(dataset) for mode, dataset in datasets.items()}
            self.datasets = datasets
            if self.verbose:
                print("number of samples in training set:", self.tvt_sizes["train"])
                print("number of samples in validation set:", self.tvt_sizes["valid"])
                print("number of samples in testing set:", self.tvt_sizes["test"])

            # generate dataloaders
            dataloaders = {
                mode: self._gen_dataloader(dataset, mode)
                for mode, dataset in datasets.items()
            }
            self.dl_train = dataloaders["train"]
            self.dl_valid = dataloaders["valid"]
            self.dl_test = dataloaders["test"]
        def _init_meta_dataloaders(self, dataset_type: type) -> None:
            """Initialize dataloaders
            
            Parameters
            ----------
            dataset_type : type[Dataset]
                Dataset type
            """

            # generate datasets
            mode_choice = ['train', 'valid', 'test']
            start_index = [0, len(self.tvt_dates['train']), len(self.tvt_dates['train']) + len(self.tvt_dates['valid'])]
            end_index = [len(self.tvt_dates['train']), len(self.tvt_dates['train']) + len(self.tvt_dates['valid']), len(self.tvt_dates['train']) + len(self.tvt_dates['valid']) + len(self.tvt_dates['test'])]
            start_index = dict(zip(mode_choice, start_index))
            end_index = dict(zip(mode_choice, end_index))
            
            datasets = {
                mode: dataset_type(
                    data_path=self.data_path,
                    dates=dates,
                    times=self.times,
                    x_keep=self.x_keep,
                    y_keep=self.y_keep,
                    **self.input_shape,
                    start=start_index[mode],
                    end=end_index[mode],
                    stocknum=self.stocknum,
                    clip=self.clip,
                    extending=self.extending,
                    sequence=self.sequence,
                    time_range=list(range(130,230))
                ) for mode, dates in self.tvt_dates.items()
            }

            self.meta_tvt_sizes = {mode: len(dataset) for mode, dataset in datasets.items()}
            self.meta_datasets = datasets
            if self.verbose:
                print("number of samples in training set:", self.meta_tvt_sizes["train"])
                print("number of samples in validation set:", self.meta_tvt_sizes["valid"])
                print("number of samples in testing set:", self.meta_tvt_sizes["test"])

            # generate dataloaders
            dataloaders = {
                mode: self._gen_dataloader(dataset, mode)
                for mode, dataset in datasets.items()
            }
            self.meta_train = dataloaders["train"]
            self.meta_valid = dataloaders["valid"]
            self.meta_test = dataloaders["test"]
        def _init_optimizer(self, optimizer_type: type, keys) -> None:
            """Initialize optimizer
            
            Parameters
            ----------
            optimizer_type : type[Optim.Optimizer]
                Optimizer type
            keys : list
                Model keys
            """

            self.optimizer_dict = {}
            # 2D params (matmul, embedding) decay but 1D params (bias, Layernorm) don't
            for key in keys:
                model = self.model_dict[key]
                grad_params = [p for p in model.parameters() if p.requires_grad]
                if key == 'model' and self.scheme == 'tra':
                    grad_params = [p for p in self.tra.parameters() if p.requires_grad]
                decay_params = [p for p in grad_params if p.dim() >= 2]
                nondecay_params = [p for p in grad_params if p.dim() < 2]
                optim_groups = [
                    {"params": decay_params, "weight_decay": self.weight_decay},
                    {"params": nondecay_params, "weight_decay": 0.0},
                ]
                if self.verbose:
                    n_decay_params = sum(p.numel() for p in decay_params)
                    n_nondecay_params = sum(p.numel() for p in nondecay_params)
                    lines = f"{key}\n"
                    lines += f"# number of decayed tensors {len(decay_params)}, "
                    lines += f"with {n_decay_params} parameters \n"
                    lines += f"# number of non-decayed tensors {len(decay_params)}, "
                    lines += f"with {n_nondecay_params} parameters"
                    print(lines)
                
                # inspect optimizer
                extra_args = {}
                available_params = inspect.signature(optimizer_type).parameters
                if "betas" not in available_params and self.verbose:
                    print(f"{optimizer_type.__name__} has no `betas`")
                else:
                    extra_args["betas"] = self.betas
                self.optimizer_dict[key] = optimizer_type(optim_groups, lr=self.lr, **extra_args, fused=False)
            
            if self.meta:
                self.optimizer_dict["meta"] = torch.optim.AdamW(self.vnet.parameters(), lr=1e-3, weight_decay=1e-4)

        def init_optimizer(self, model, optimizer_type: type) -> None:
            """Initialize optimizer

            2D params (matmul, embedding) decay but 1D params (bias, Layernorm) don't
            """
            grad_params = [p for p in model.parameters() if p.requires_grad]
            decay_params = [p for p in grad_params if p.dim() >= 2]
            nondecay_params = [p for p in grad_params if p.dim() < 2]
            optim_groups = [
                {"params": decay_params, "weight_decay": self.weight_decay},
                {"params": nondecay_params, "weight_decay": 0.0},
            ]

            # inspect optimizer
            extra_args = {}
            available_params = inspect.signature(optimizer_type).parameters
            if "betas" not in available_params and self.verbose:
                print(f"{optimizer_type.__name__} has no `betas`")
            else:
                extra_args["betas"] = self.betas

            return optimizer_type(optim_groups, lr=self.lr, **extra_args)

        def _init_scheduler(
            self,
            warmup_type: type,
            anneal_type: type,
            keys: list
        ) -> None:
            """Initialize learning rate scheduler

            Parameters
            ----------
            warmup_type : type[LRScheduler]
                Warmup scheduler type
            anneal_type : type[LRScheduler]
                Anneal scheduler type
            keys : list
                List of model keys
            """
            # warmup
            self.warmup_group, self.anneal_group = [], []
            T_warmup = self.warmup_end * len(self.dl_train) // self.sub_epochs
            for key in keys:
                warmup = warmup_type(
                    self.optimizer_dict[key],
                    lr_lambda=lambda t: self.warmup_base ** (t / T_warmup) / self.warmup_base,
                )
                # anneal
                T_anneal = self.sub_epochs * self.epochs - self.anneal_start
                anneal = anneal_type(
                    self.optimizer_dict[key],
                    T_max=T_anneal,
                    eta_min=self.anneal_eta_min
                )
                self.warmup_group.append(warmup)
                self.anneal_group.append(anneal)
        def _init_backtester(self, backtester_type: type) -> None:
            """Initialize backtester

            Parameters
            ----------
            backtester_type : type[Backtester]
                Backtester type
            """
            self.backtester = backtester_type(
                self.n_thresh,
                self.bt_eps,
                self.pct_l,
                self.pct_s,
                self.n_groups,
                self.scale,
                len(self.y_keep),
            )

        def calculate_quantiles(self, tensor):
            sorted_tensor, indices = torch.sort(tensor)
            ranks = torch.argsort(indices).float()
            quantiles = ranks / (len(tensor) - 1)
            return quantiles
        def bin_and_replace_with_mean(self, tensor, num_bins):
            min_val, max_val = tensor.min(), tensor.max()
            bin_width = (max_val - min_val) / num_bins
            bins = (((tensor - min_val) / bin_width).floor().clamp(0, num_bins - 1)).long()
            bin_sums = torch.zeros(num_bins, device=tensor.device)
            bin_counts = torch.zeros(num_bins, device=tensor.device)
            bin_sums.scatter_add_(0, bins, tensor)
            bin_counts.scatter_add_(0, bins, torch.ones_like(tensor))
            bin_means = bin_sums / bin_counts.clamp(min=1)
            binned_tensor = bin_means[bins]
            return binned_tensor

        def weight_map(self, cur_loss, init_loss, num_bins, alpha=0.5):
            org_shape = cur_loss.shape
            cur_loss = cur_loss.reshape(-1)
            init_loss = init_loss.reshape(-1)
            h1 = self.calculate_quantiles(-cur_loss)
            h2 = self.calculate_quantiles(cur_loss / init_loss)
            h = alpha * h1 + (1 - alpha) * h2
            w = 1 / (0.5 * h + 0.1)
            w = self.bin_and_replace_with_mean(w, num_bins=num_bins)
            w = w.view(*org_shape)
            return w

        def step_calc_loss(self, model, data: tuple, weight=None, **kwargs) -> torch.Tensor:
            x, y_true, pad_mask, ignore_index, time = data
            signal = kwargs.get('signal')
            pos_index = y_true > 0
            if signal == 'class':
                class_pred = model(x, pad_mask=pad_mask, signal='class')
                y_binary = (y_true > 0).long()
                criterion = nn.CrossEntropyLoss()
                binary_loss = criterion(class_pred[~ignore_index].view(-1, class_pred.shape[-1]), y_binary[~ignore_index].view(-1))
                return binary_loss, pos_index
            elif signal == 'pos':
                pos_pred = model(x[pos_index], pad_mask=pad_mask, signal='pos')
                pos_loss = F.mse_loss(pos_pred, y_true[pos_index])
                return pos_loss, pos_pred
            else:
                neg_pred = model(x[~pos_index], pad_mask=pad_mask, signal='neg')
                neg_loss = F.mse_loss(neg_pred, y_true[~pos_index])
                return neg_loss, neg_pred
            return torch.zeros(y_true.shape[0]).to(y_true), pos_loss, neg_loss, y_pred
        def tra_calc_loss(self, model, data: tuple, weight=None, **kwargs):
            e = kwargs.get('e')
            x, y_true, pad_mask, ignore_index, time = data
            hidden = model(x, pad_mask=pad_mask)
            y_pred, all_preds ,prob= self.tra(hidden, time)
            y_true = y_true.view(-1)
            if weight is not None:
                loss = torch.mean(weight * (y_true - y_pred) ** 2)
            else:
                loss = torch.mean((y_true - y_pred) ** 2)
            L = (y_pred - y_true).pow(2).mean()
            L = (all_preds.detach() - y_true[:, None]).pow(2)
            L = -L.min(dim=1, keepdim=True).values  # normalize & ensure positive input

            if prob is not None:
                P = sinkhorn(-L, epsilon=0.01)  # sample assignment matrix
                lamb = self.lamb * (self.rho ** e)
                reg = P.log().mul(P).sum(dim=-1).mean()
                loss = loss - lamb * reg
            return loss, L.reshape(self.train_batch_size, self.N, self.num_states), y_pred.reshape(self.train_batch_size, -1)

        def naive_calc_loss(self, model, data: tuple, weight=None, **kwargs) -> torch.Tensor:
            x, y_true, pad_mask, ignore_index, time = data
            y_pred = model(x, pad_mask=pad_mask)
            if weight is not None:
                loss = torch.mean(weight * (y_true - y_pred) ** 2)
            else:
                loss = torch.mean((y_true - y_pred) ** 2)

            sample_loss = (y_true - y_pred) ** 2
            loss = F.mse_loss(y_pred[~ignore_index], y_true[~ignore_index])
            return loss, sample_loss, y_pred

        def switch_calc_loss(self, model, data: tuple, weight=None, **kwargs) -> torch.Tensor:
            x, y_true, pad_mask, ignore_index, time = data
            y_pred, aux = model(x, pad_mask=pad_mask)
            if weight is not None:
                loss = torch.mean(weight * (y_true - y_pred) ** 2) + torch.mean(aux)
            else:
                loss = torch.mean((y_true - y_pred) ** 2) + torch.mean(aux)

            sample_loss = (y_true - y_pred) ** 2
            loss = F.mse_loss(y_pred[~ignore_index], y_true[~ignore_index])
            return loss, sample_loss, y_pred
        def naive_prep_eval(self, data: tuple) -> tuple:
            # e.g.
            model = self.model_dict['model']
            x, y_true, pad_mask, ignore_index, time = data
            y_pred = model(x, pad_mask=pad_mask)
            if y_pred.ndim == 3:
                y_pred = y_pred[..., 1]
            if y_true.ndim == 3:
                y_true = y_true[..., 1]
            return y_true, y_pred, ignore_index

        def tra_prep_eval(self, data: tuple) -> tuple:
            tra, model = self.model_dict['tra'], self.model_dict['model']
            x, y_true, pad_mask, ignore_index, time = data
            hidden = model(x, pad_mask=pad_mask)
            y_pred, all_preds, prob = tra(hidden, time)
            L = (all_preds - y_true.view(-1, self.N)[:, None]).pow(2)
            L = -L.min(dim=1, keepdim=True).values
            return y_pred.view(-1, self.N), y_true.view(-1, self.N), ignore_index, L.reshape(self.valid_batch_size, self.N, self.num_states)

        def set_model(
            self, model_type: type, 
            model_config, 
            vnet=None, 
            tra=None, 
            *path_args
        ) -> None:
            """Set model
            
            Parameters
            ----------
            model_type : type[nn.Module]
                Model type
            state_dict_path : str | None, optional
                Path of model state, by default None
            """
            model_path = list(path_args)
            model_config['L'] = self.input_shape['L']
            model_config['C'] = self.input_shape['C']
            self.model_dict = {
                'model': model_type(*model_path, **model_config)
            }
          
            if self.meta:
                assert vnet is not None
                self.model_dict['vnet'] = self._to_device(vnet)
            if self.scheme == 'tra':
                assert tra is not None
                self.model_dict['tra'] = self._to_device(tra)
            self.model_dict['model'] = self._to_device(model_type(model_config))
            for path in model_path:
                name = path.split("_")[0]
                state_dict = torch.load(path)
                state_dict = {k: v for k, v in zip(self.model_dict[name].state_dict().keys(), state_dict.values())}
                self.model_dict[name].load_state_dict(state_dict)

            if self.verbose:
                for key, model in self.model_dict.items():
                    print("model architecture:", model)
                    n_params = sum(p.numel() for p in model.parameters())
                    print("number of model parameters:", n_params)

        def build(
            self,
            dataset_type: type = MmapDataset,
            optimizer_type: type = optim.AdamW,
            warmup_type: type = LambdaLR,
            anneal_type: type = CosineAnnealingLR,
            backtester_type: type = Backtester,
            *optimizer_dict_path
        ) -> None:
            """Build pipeline

            Parameters
            ----------
            dataset_type : type[Dataset], optional
                Dataset type, by default MmapDataset
            optimizer_type : type[optim.Optimizer], optional
                Optimizer type, by default AdamW
            warmup_type : type[LRScheduler], optional
                Warmup scheduler type, by default LambdaLR
            anneal_type : type[LRScheduler], optional
                Anneal scheduler type, by default CosineAnnealingLR
            backtester_type : type[Backtester], optional
                Backtester type, by default Backtester
            """
            # dataloaders
            self._init_dataloaders(dataset_type)


# optimizer
            keys = ['model']
            self._init_optimizer(optimizer_type, keys)
            self._init_scheduler(warmup_type, anneal_type, keys)
            opm_path = list(*optimizer_dict_path)
            for path in opm_path:
                name = path.split("_")[0]
                self.optimizer_dict[name].load_state_dict(torch.load(optimizer_dict_path))

            # backtester
            self._init_backtester(backtester_type)

        def train(self, start=0) -> None:
            """Train model"""
            loss_epoch = []
            # hist = torch.zeros(((len(self.tvt_dates['train'])+1, self.S, self.N, self.num_states))
            # hist=torch.Load('model/base_WITRAN_tra_1/base_WITRAN_tra_1_train_0-5_Loss.pt')
            for e in range(start, self.epochs):
                if e == 1: break
                # training mode
                loss_cur = []
                yp_list = []
                for model in self.model_dict.values():
                    model.train()
                
                # DDP shuffle
                if self.ddp:
                    self.dl_train.sampler.set_epoch(e)
                    if self.meta:
                        self.meta_train.sampler.set_epoch(e)
                        self.meta_iter = iter(self.meta_train)
                    shuffle_index = torch.tensor(list(self.dl_train.sampler.__iter__())).to(self.local_rank)

                if self.scheme == 'de':
                    if e >= 2:
                        cur_weight = self.weight_map(cur_loss=loss_epoch[-1], init_loss=loss_epoch[0], num_bins=1000)
                        cur_weight = cur_weight * cur_weight.numel() / cur_weight.sum()
                        cur_weight = cur_weight.reshape(-1, self.stocknum).to(self.local_rank)
                    else:
                        cur_weight = None

                for b, data in tqdm(enumerate(self.dl_train)):
                    # move tensors to correct device
                    cur_index = shuffle_index[b * self.train_batch_size : min((b+1)*self.train_batch_size, len(shuffle_index))]
                    if self.ddp:
                        data = [i.to(self.local_rank) for i in data]

                    else:
                        data = [i.to(self.device) for i in data]

                    if self.meta:
                        try:
                            meta_data = next(self.meta_iter)
                        except StopIteration:
                            self.meta_iter = iter(self.meta_train)
                            meta_data = next(self.meta_iter)
                        if self.ddp:
                            meta_data = [i.to(self.local_rank) for i in meta_data]
                        else:
                            meta_data = [i.to(self.device) for i in meta_data]

                    model, vnet, optimizer, vnet_optimizer = self.model_dict['model'], self.model_dict['meta'], self.optimizer_dict['model'], self.optimizer_dict['meta']
                    temp_model = copy.deepcopy(model).to(self.local_rank)
                    temp_optimizer = self.init_optimizer(temp_model, optim.AdamW)
                    temp_optimizer.load_state_dict(optimizer.state_dict())
                    loss, sample_loss, _ = self.naive_calc_loss(temp_model, data, weight=None)
                    loss = loss.unsqueeze(-1)
                    v_weight = vnet(loss).squeeze()
                    v_weight = v_weight * v_weight.numel() / v_weight.sum()
                    loss = loss * v_weight
                    temp_optimizer.zero_grad()
                    loss.backward()
                    temp_optimizer.step()

                    meta_loss, _, _ = self.naive_calc_loss(temp_model, meta_data)
                    vnet_optimizer.zero_grad()
                    meta_loss.backward()
                    vnet_optimizer.step()

                    del temp_model, temp_optimizer

                    optimizer.zero_grad()
                    loss, _, _ = self.loss_func(self.model, data)
                    loss = loss.unsqueeze(-1)
                    with torch.no_grad():
                        v_weight = vnet(loss).squeeze()
                        v_weight = v_weight * v_weight.numel() / v_weight.sum()
                    loss, sample_loss, y_pred = self.naive_calc_loss(model, data, weight=v_weight)
                    # cur_index=[i//self.S for i in cur_index]
                    # hist_Loss=hist[cur_index].transpose(1,2).reshape(-1,self.S,self.num_states) (tra)
                    model, optimizer = self.model_dict['model'], self.optimizer_dict['model']
                    weight = cur_weight[cur_index] if self.scheme == 'de' and e >= 2 else None
                    optimizer.zero_grad()
                    loss, sample_loss, y_pred = self.loss_func(model, data=data, weight=weight)
                    loss_cur.append(sample_loss.detach())
                    loss.backward()
                    optimizer.step()

                    # naive
                    # elif self.scheme=='step':
                    #     y_pred=torch.zeros(data[1].shape).to(self.Local_rank)
                    #     class_Loss,pos_index=self.step_calc_Loss(data,signal='class')
                    #     self.optimizer.zero_grad()
                    #     class_Loss.backward()
                    #     self.optimizer.step()
                    #     pos_Loss,pos_pred=self.step_calc_Loss(data,signal='pos')
                    #     self.optimizer.zero_grad()
                    #     pos_Loss.backward()
                    #     self.optimizer.step()
                    #     neg_Loss,neg_pred=self.step_calc_Loss(data,signal='neg')
                    #     self.optimizer.zero_grad()
                    #     neg_Loss.backward()
                    #     self.optimizer.step()
                    #     y_pred[pos_index]=pos_pred.clone().detach()
                    #     y_pred[~pos_index]=neg_pred.clone().detach()
                    torch.cuda.empty_cache()
                    for name, param in self.model.named_parameters():
                        if param.grad is not None:
                            grad_norm = param.grad.norm().item()
                            self.writer.add_scalar(f'GradNorm/{name}', grad_norm, b)
                        if b % 1000 == 0:
                            print(grad_norm)

                    # check sub epoch
                    yp_list.append(y_pred.detach())
                    batches_per_sub_epoch = len(self.dl_train) // self.sub_epochs
                    is_at_sub_e = ((b + 1) % batches_per_sub_epoch) == 0
                    sub_e = (b + 1) // batches_per_sub_epoch
                    sub_e_passed = e * self.sub_epochs + sub_e
                    # scheduler update
                    is_at_sub_e = True
                    if sub_e_passed < self.warmup_end:
                        for warmup in self.warmup_group:
                            warmup.step()
                    if is_at_sub_e and sub_e_passed >= self.anneal_start:
                        for anneal in self.anneal_group:
                            anneal.step()

                    # save model
                    if not self.local_rank and is_at_sub_e:
                        for key, model in self.model_dict.items():
                            torch.save(
                                model.state_dict(),
                                self.dir_m_exp + f"{self.model_name}_{key}_{e}-{sub_e}.pt",
                            )
                        for key, optimizer in self.optimizer_dict.items():
                            torch.save(
                                optimizer.state_dict(),
                                self.dir_m_exp + f"{self.model_name}_{key}_{e}-{sub_e}_optimizer.pt",
                            )

                    # validate model
                    if self.enable_valid and is_at_sub_e:
                        self.evaluate("valid", True, f"{e}-{sub_e}", True, e, sub_e)
                        self.evaluate("test", True, f"{e}-{sub_e}", True, e, sub_e)
                        for model in self.model_dict.values():
                            model.train()

                    if self.scheme == "de":
                        loss_cur = torch.cat(loss_cur).to(self.local_rank)
                        yp = torch.cat(yp_list)
                        del yp_list
                        mode = "train"

                    if self.ddp and not self.local_rank:
                        if self.scheme == "de":
                            loss_cur = self.dist_gather(loss_cur, mode)

                    # if self.scheme=='tra' and not self.Local_rank:
                    #     shuffle_index=shuffle_index.unsqueeze(-1).unsqueeze(-1).expand(loss_cur.shape).contiguous()
                    #     loss_cur=torch.gather(input=loss_cur,dim=0,index=shuffle_index).contiguous()
                    #     shuffle_index=self.dist_gather(shuffle_index,mode)
                    #     yp=self.dist_gather(yp,mode)

                    # if self.scheme=='tra' and not self.Local_rank:
                    #     hist=hist.index_base+1,self.S,self.N,self.num_states).to(self.Local_rank)
                    # hist=loss_cur.reshape(len(self.tvt_dates[mode])-self.index_base,self.S,self.N,self.num_states)
                    # hist=torch.cat((temp,hist),dim=0).contiguous()
                    # dist.broadcast(hist,src=0)
                    # print(self.local_rank,hist.shape)
                    # if not self.local_rank:
                    #     pred_shuffle_index=shuffle_index.unsqueeze(-1).expand(yp.shape).contiguous()
                    #     yp=torch.gather(input=yp,dim=0,index=pred_shuffle_index)
                    #     a,b=yp.shape
                    #     groupnum=int(np.ceil(self.S/self.N/self.stocknum))
                    #     yp=yp.reshape(a//groupnum,b*groupnum)
                    #     stockslice=List(range(self.N//self.stocknum*self.S//self.stocknum))
                    #     stockslice.extend(List(range(b*groupnum-self.N%self.stocknum,b*groupnum)))
                    #     yp=yp[:,stockslice]
                    #     torch.save(yp,self.dir_r_exp + f'{self.model_name}_{mode}_{e}_{pred.pt}')
                    loss_epoch.append(loss_cur)
                    self.writer.close()

        @torch.no_grad()
        def evaluate(self, mode: str, save: bool = False, save_suffix: str = "", save_res=False, e=0, sub_e=0) -> None:
            """Evaluate model"""
            groupnum = int(np.ceil(self.S/self.N/self.stocknum))
            loss_cur = []
            if mode == "valid":
                dl_eval = self.dl_valid
            elif mode == "test":
                dl_eval = self.dl_test
            elif mode == 'train':
                self.dl_train=self.gen_dataloader(self.datasets['train'], mode='test')
                dl_eval = self.dl_train
            else:
                raise ValueError(f"{mode} is not supported")
            for model in self.model_dict.values():
                model.eval()
            # synchronize processes
            if self.ddp:
                dist.barrier()
            shuffle_index=torch.tensor(list(dl_eval.sampler.__iter__())).to(self.local_rank)

            # evaluate
            yt_list, yp_list, ii_list = [], [], []
            hist=torch.zeros((len(self.tvt_dates[mode])+1,self.S,self.N,self.num_states)).to(self.local_rank)

            for index, data in tqdm(enumerate(dl_eval)):
                cur_index=shuffle_index[index*self.valid_batch_size:min((index+1)*self.valid_batch_size,len(shuffle_index))]
                # move tensors to correct device
                if self.ddp:
                    data = [i.to(self.local_rank) for i in data]
                else:
                    data = [i.to(self.device) for i in data]
                
                # process batch
                if self.scheme == 'stack':
                    yt, yp, ii = self.stack_prep_eval(data)
                elif self.scheme == 'tra':
                    # temp=[i//self.S for i in cur_index]
                    # hist_Loss=hist[temp].transpose(1,2).reshape(-1,self.S,self.num_states)
                    yp,yt,ii,sample_Loss=self.tra_prep_eval(data)
                    # cur_index=self.dist_gather(cur_index,mode)
                    # sample_Loss=self.dist_gather(sample_Loss,mode)
                    # date=[i//self.S+self.index_base+1 for i in cur_index]
                    # time=[i%self.S for i in cur_index]
                    # hist[date,time]=sample_Loss
                    # dist.broadcast(hist,src=0)
                else:
                    yt, yp, ii=self.naive_prep_eval(data)
                yt_list.append(yt)
                yp_list.append(yp)
                ii_list.append(ii)

            # concatenate across batches
            yt = torch.cat(yt_list)
            yp = torch.cat(yp_list)
            ii = torch.cat(ii_list)
            print(self.local_rank, yp.shape)
            # print(self.local_rank,yp.shape)
            if self.ddp:
                yt = self._dist_gather(yt, mode)
                yp = self._dist_gather(yp, mode)
                ii = self._dist_gather(ii, mode)
            # if not self.local_rank:
            #     torch.save(hist,self.dir.m_exp + f'{self.model_name}_{mode}_{e}-{sub_e}_Loss.pt')
            # if self.sequence:
            #     yt=yt.view(-1,self.N)
            #     yp=yp.view(-1,self.N)
            #     ii=ii.view(-1,self.N)

            if save_res and not self.local_rank:
                a,b=yt.shape
                groupnum=int(np.ceil(self.N/self.stocknum))
                yt,yp,ii=yt.reshape(a//groupnum,b*groupnum),yp.reshape(a//groupnum,b*groupnum),ii.reshape(a//groupnum,b*groupnum)
                stockslice=list(range(self.N//self.stocknum*self.stocknum))
                stockslice.extend(list(range(b*groupnum-self.N%self.stocknum,b*groupnum)))
                yt,yp,ii=yt[:,stockslice],yp[:,stockslice],ii[:,stockslice]
                torch.save(yp,self.dir.m_exp + f'{self.model_name}_{mode}_{e}-{sub_e}_pred.pt')
            if not self.local_rank:  # None for DP and 0 for DDP

                metrics = self.backtester.run(yt, yp, ii)
                metrics_dict = {k: v.cpu().numpy() for k, v in zip(self.metrics, metrics)}
                # save
                if save_res:
                    save_suffix = f"_{save_suffix}" if len(save_suffix) else ""
                    filename = f"{self.model_name}_{mode}{save_suffix}.npz"
                    np.savez_compressed(self.dir.r_exp + filename, **metrics_dict)
                    self.backtester.show(self.dir.r_exp, filename)










                    


