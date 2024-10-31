

import sys
sys.path.append('/dfs/data/package')
import inspect
import os
import random
from abc import ABC, abstractmethod
from torch.profiler import profile, record_function, ProfilerActivity
import torchvision.models as models
import numpy as np
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
from .data import MmapDataset,StockMmapDataset,IntraMmapDataset
from .utils import Directory, get_tvt_dates
from tensorboardX import SummaryWriter
from .VNet.VNet import VNet
from prefetch_generator import BackgroundGenerator
torch.autograd.set_detect_anomaly(True)
os.environ['TORCH_USE_CUDA_DSA'] = "1"
class DataLoaderX(DataLoader):

    def __iter__(self):
        return BackgroundGenerator(super().__iter__())
def shoot_infs(inp_tensor):
    """Replaces inf by maximum of tensor"""
    mask_inf = torch.isinf(inp_tensor)
    ind_inf = torch.nonzero(mask_inf, as_tuple=False)
    if len(ind_inf) > 0:
        for ind in ind_inf:
            if len(ind) == 2:
                inp_tensor[ind[0], ind[1]] = 0
            elif len(ind) == 1:
                inp_tensor[ind[0]] = 0
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

    def __init__(
        self,
        dir: Directory,
        configs,
        verbose: bool = False,
    ) -> None:
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
#         with open(dir.config + configs_filename) as f:
#             configs = yaml.safe_load(f)
        configs=configs
        save_suffix = configs["prep"]["save_suffix"]
        save_suffix = f"_{save_suffix}" if len(save_suffix) else ""
        data_info_filename = f"data_info{save_suffix}.npz"
        data_info = np.load(dir.config + data_info_filename, allow_pickle=True)
        # data
        self.data_path = str(data_info["path"])
#         self.data_path='/dfs/data/data/tc/'
        self.dates = data_info["dates"]
        self.times = data_info["times"]
        self.tickers = data_info["tickers"]
        self.partition = configs["data"]["partition"]
        self.x_keep = [i for i in range(241) if i not in configs["data"]["x_skip"]]
        self.y_keep = [i for i in range(241) if i not in configs["data"]["y_skip"]]
        self.clip = configs["data"]["clip"]
        self.writer=SummaryWriter(logdir=configs['model']['writer'])
        self.extending = configs["data"]["extending"]
        self.stocknum=configs['data']['stocknum']
        self.sequence=configs['data']['sequence']
                # seed
        self.S=len(self.y_keep)
        self.input_shape = configs["model"]["input_shape"]
        self.index_base = self.input_shape['L'] // self.S + 1
        self.tvt_dates = get_tvt_dates(self.dates, self.partition)
        # optim
        self.seed = configs["optim"]["seed"]
        self.train_batch_size = configs["optim"]["train_batch_size"]
        self.valid_batch_size = configs["optim"]["valid_batch_size"]
        self.num_workers = configs["optim"]["num_workers"]
        self.ddp = configs["optim"]["ddp"]
        self.lr = configs["optim"]["lr"]
        self.betas = configs["optim"]["betas"]
        self.weight_dacay = configs["optim"]["weight_decay"]
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
        self.pct_l = configs["backtest"]["pct_l"]
        self.pct_s = configs["backtest"]["pct_s"]
        self.n_groups = configs["backtest"]["n_groups"]
        self.scale = configs["backtest"]["scale"]
        self.metrics = configs["backtest"]["metrics"]
        self.rho=0.99
        self.lamb=2
        
        # general
        dir_m_exp = dir.model + self.model_name + "/"
        dir_r_exp = dir.result + self.model_name + "/"
        dir.add_dirs(m_exp=dir_m_exp, r_exp=dir_r_exp)
        self.dir = dir
        self.verbose = verbose
        self.N=configs['model']['input_shape']['N']

        self.num_states=5
        self._set_seed()
        # device
        self._set_device()
        # scheme
        self.model_keys=['class','pos','neg']
        self.scheme=configs['scheme']['type']
        self.meta=configs['scheme']['meta']
        self.vae=configs['scheme']['vae']
        try:
            self.loss_func=getattr(self,self.scheme+'_calc_loss')
        except:
            self.loss_func=getattr(self,'naive_calc_loss')
#         self.eval_func=getattr(self,self.scheme+'_prep_eval')
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
            self.world_size = None
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

    def _to_device(self,model) -> None:
        """Model to device"""
        if self.ddp:  # DDP
            model = model.to(self.local_rank)
            model = DistributedDataParallel(
                module=model,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                find_unused_parameters=True,
            )
        else:  # DP
            model.to(self.device)
#             model = nn.DataParallel(model)
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
            batch_size=self.train_batch_size if mode=="train" else self.valid_batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=True,
            
#             collate_fn=self.collate_fn if isinstance(dataset,StockMmapDataset) else None,
            drop_last=True if mode == "train" else False,
        )
        return dataloader
    
    
    def _dist_gather(self, t: torch.Tensor, mode: str,local_rank=0) -> torch.Tensor:
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
#         print('t_shape:',t.shape)
        all_t = [torch.empty_like(t) for _ in range(self.world_size)]
        # gather tensors to rank 0
        dist.gather(t, all_t if self.local_rank == local_rank else None, local_rank)
        # process on rank 0
        if self.local_rank == local_rank:
            # adjust to correct sample order
            order, shape = (1, 0, *range(2, t.ndim + 1)), (-1, *t.shape[1:])
            t = torch.stack(all_t).permute(order).reshape(shape)
            # drop duplicated samples created by DDP
            t = t[:self.tvt_sizes[mode]]
        return t

    def _init_dataloaders(self, dataset_type: type) -> None:
        """Initialize dataloaders

        Parameters
        ----------
        dataset_type : type[Dataset]
            Dataset type
        """
        # generate datasets
        mode_choice=['train','valid','test']
        start_index=[0,len(self.tvt_dates['train']),len(self.tvt_dates['train'])+len(self.tvt_dates['valid'])]
        start_index=dict(zip(mode_choice,start_index))
        end_index=[len(self.tvt_dates['train']),len(self.tvt_dates['train'])+len(self.tvt_dates['valid']),len(self.tvt_dates['train'])+len(self.tvt_dates['valid'])+len(self.tvt_dates['test'])]
        end_index=dict(zip(mode_choice,end_index))
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
            )
            for mode, dates in self.tvt_dates.items()
        }
#         self.vnet_loader=
        self.tvt_sizes = {mode: len(dataset) for mode, dataset in datasets.items()}
        self.datasets=datasets
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
        mode_choice=['train','valid','test']
        start_index=[0,len(self.tvt_dates['train']),len(self.tvt_dates['train'])+len(self.tvt_dates['valid'])]
        start_index=dict(zip(mode_choice,start_index))
        end_index=[len(self.tvt_dates['train']),len(self.tvt_dates['train'])+len(self.tvt_dates['valid']),len(self.tvt_dates['train'])+len(self.tvt_dates['valid'])+len(self.tvt_dates['test'])]
        end_index=dict(zip(mode_choice,end_index))
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
                time_range=list(range(100,238))
                
            )
            for mode, dates in self.tvt_dates.items()
        }
        self.meta_tvt_sizes = {mode: len(dataset) for mode, dataset in datasets.items()}
        self.meta_datasets=datasets
        if self.verbose:
            print("number of samples in meta_training set:", self.meta_tvt_sizes["train"])
            print("number of samples in meta_validation set:", self.meta_tvt_sizes["valid"])
            print("number of samples in meta_testing set:", self.meta_tvt_sizes["test"])
        # generate dataloaders
        dataloaders = {
            mode: self._gen_dataloader(dataset, mode)
            for mode, dataset in datasets.items()
        }
        self.meta_train = dataloaders["train"]
        self.meta_valid = dataloaders["valid"]
        self.meta_test = dataloaders["test"]
    
    def _init_optimizer(self, optimizer_type: type,keys) -> None:
        """Initialize optimizer

        Parameters
        ----------
        optimizer_type : type[optim.Optimizer]
            Optimizer type
        """
        self.optimizer_dict={}
        # 2D params (matmul, embedding) decay but 1D params (bias, layernorm) don't
        for key in keys:
            model=self.model_dict[key]
            grad_params = [p for p in model.parameters() if p.requires_grad]
            if key=='model' and self.scheme=='tra':
                grad_params=grad_params+[p for p in self.tra.parameters() if p.requires_grad]
            decay_params = [p for p in grad_params if p.dim() >= 2]
            nondecay_params = [p for p in grad_params if p.dim() < 2]
            optim_groups = [
                {"params": decay_params, "weight_dacay": self.weight_dacay},
                {"params": nondecay_params, "weight_dacay": 0.0},
            ]
            if self.verbose:
                n_decay_params = sum(p.numel() for p in decay_params)
                n_nondecay_params = sum(p.numel() for p in nondecay_params)
                lines = f"{key}"
                lines += f"number of decayed tensors {len(decay_params)}, "
                lines += f"with {n_decay_params} parameters \n"
                lines += f"number of non-decayed tensors {len(decay_params)}, "
                lines += f"with {n_nondecay_params} parameters"
                print(lines)
            # inspect optimizer
            extra_args = {}
            available_params = inspect.signature(optimizer_type).parameters
            if "betas" not in available_params and self.verbose:
                print(f"{optimizer_type.__name__} has no `betas`")
            else:
                extra_args["betas"] = self.betas
            self.optimizer_dict[key]=optimizer_type(optim_groups, lr=self.lr, **extra_args,fused=False)
        if self.meta:
            grad_params = [p for p in self.model_dict['meta'].parameters() if p.requires_grad]
            decay_params = [p for p in grad_params if p.dim() >= 2]
            nondecay_params = [p for p in grad_params if p.dim() < 2]
            meta_optim_groups = [
                {"params": decay_params, "weight_dacay": self.weight_dacay},
                {"params": nondecay_params, "weight_dacay": 0.0},
            ]
            
            self.optimizer_dict['meta']=torch.optim.AdamW(meta_optim_groups, 1e-3,weight_decay=1e-4)

    
    def init_optimizer(self,model, optimizer_type: type) -> None:
        # 2D params (matmul, embedding) decay but 1D params (bias, layernorm) don't
        grad_params = [p for p in model.parameters() if p.requires_grad]
        decay_params = [p for p in grad_params if p.dim() >= 2]
        nondecay_params = [p for p in grad_params if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_dacay": self.weight_dacay},
            {"params": nondecay_params, "weight_dacay": 0.0},
        ]
        # inspect optimizer
        extra_args = {}
        available_params = inspect.signature(optimizer_type).parameters
        if "betas" not in available_params:
            print(f"{optimizer_type.__name__} has no `betas`")
        else:
            extra_args["betas"] = self.betas
        return optimizer_type(optim_groups, lr=self.lr, **extra_args)
    
    def pace_weight(self,loss,a=1e-3,b=1e-4):
        return torch.minimum(torch.maximum(torch.tensor(0),torch.tensor(1)-(loss-b)/a),torch.tensor(1))
    def _init_scheduler(
        self,
        warmup_type: type,
        anneal_type: type,
        keys:list
    ) -> None:
        """Initialize learning rate scheduler

        Parameters
        ----------
        warmup_type : type[LRScheduler]
            Warmup scheduler type
        anneal_type : type[LRScheduler]
            Anneal scheduler type
        """
        # warmup
        self.warmup_group,self.anneal_group=[],[]
        T_warmup = self.warmup_end * len(self.dl_train) // self.sub_epochs
        for key in keys:
            warmup = warmup_type( self.optimizer_dict[key],lr_lambda=lambda t: self.warmup_base ** (t / T_warmup) / self.warmup_base,)
            # anneal
            T_anneal = self.sub_epochs * self.epochs - self.anneal_start
            anneal = anneal_type(self.optimizer_dict[key], T_max=T_anneal, eta_min=self.anneal_eta_min)
            self.warmup_group.append(warmup)
            self.anneal_group.append(anneal)
#     def collate_fn(self,batch)->tuple:
#         a_list, b_list, c_list, d_list,e_list = [], [], [], [],[]
#         for sample in batch:
#             a, b, c, d,e= sample
#             a_list.append(a)
#             b_list.append(b)
#             c_list.append(c)
#             d_list.append(d)
#             e_list.append(e)
#         # 将列表中的数据在第一个维度上合并
#         a_cat = torch.cat(a_list, dim=0) if a_list else torch.tensor([])  # 合并第一个维度
#         b_cat = torch.cat(b_list, dim=0) if b_list else torch.tensor([])
#         c_cat = torch.cat(c_list, dim=0) if c_list else torch.tensor([])
#         d_cat = torch.cat(d_list, dim=0) if d_list else torch.tensor([])

#         # 返回合并后的数据
#         return a_cat, b_cat, c_cat, d_cat
    
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
      sorted_tensor, indices = torch.sort(tensor)
        ranks = torch.argsort(indices).float()
        quantiles = ranks / (len(tensor) - 1)
        return quantiles
    
    def bin_and_replace_with_mean(self,tensor, num_bins):
    
        min_val, max_val = tensor.min(), tensor.max()
        bin_width = (max_val - min_val) / num_bins
        bins = ((tensor - min_val) / bin_width).floor().clamp(0, num_bins - 1).long()
        bin_sums = torch.zeros(num_bins, device=tensor.device)
        bin_counts = torch.zeros(num_bins, device=tensor.device)
        bin_sums.scatter_add_(0, bins, tensor)
        bin_counts.scatter_add_(0, bins, torch.ones_like(tensor))
        bin_means = bin_sums / bin_counts.clamp(min=1)
        binned_tensor = bin_means[bins]
        return binned_tensor
        
    def weight_map(self,cur_loss,init_loss,num_bins,alpha=0.5):
        org_shape=cur_loss.shape
        cur_loss=cur_loss.reshape(-1)
        init_loss=init_loss.reshape(-1)
        h1=self.calculate_quantiles(-cur_loss)
        h2=self.calculate_quantiles(cur_loss/init_loss)
        h=alpha*h1+(1-alpha)*h2
        w=1/(0.5*h+0.1)
        w=self.bin_and_replace_with_mean(w,num_bins=num_bins)
        w=w.view(*org_shape)
        return w
    def class_vae_loss(self,model,data:tuple,weight=None,**kwargs):
        x, y_true, pad_mask, ignore_index,time= data
        x_decode,mu,logvar,batch_res,prob=model(x,pad_mask=pad_mask,signal='vae')
#         print(model.module())
        return model.module.vae.loss(x_decode,x,mu,logvar,prob,batch_res)
    def vae_loss(self,model,data:tuple,weight=None,**kwargs):
        x, y_true, pad_mask, ignore_index,time= data
        loss=model(x,pad_mask=pad_mask,signal='vae')
        return loss
#         x_rec,mu,logvar=model(x,pad_mask=pad_mask,signal='vae')
#         recon_loss = F.mse_loss(x_rec, x, reduction='mean')*x.shape[-1]  # Mean Squared Error
#         kl_loss = torch.mean(-0.5 * torch.sum((logvar - mu.pow(2) - logvar.exp()),dim=-1))
#         return recon_loss + kl_loss
    def step_calc_loss(self,model, data: tuple,weight=None,**kwargs) -> torch.Tensor:
        x, y_true, pad_mask, ignore_index,time= data
        signal=kwargs.get('signal')
        pos_index=y_true>0
        if signal=='class':
            class_pred = model(x, pad_mask=pad_mask,signal='class')
            y_binary = (y_true > 0).long()
            criterion = nn.CrossEntropyLoss()
            binary_loss = criterion(class_pred[~ignore_index].view(-1,class_pred.shape[-1]), y_binary[~ignore_index].view(-1))
            return binary_loss,pos_index
        elif signal=='pos':
            pos_pred=model(x[pos_index],pad_mask=pad_mask,signal='pos')
            pos_loss=2*F.mse_loss(pos_pred,y_true[pos_index])
            return pos_loss,pos_pred
        else:
            neg_pred=model(x[~pos_index],pad_mask=pad_mask,signal='neg')
            neg_loss=F.mse_loss(neg_pred,y_true[~pos_index])
            return neg_loss,neg_pred
        y_pred=torch.zeros(y_true.shape).to(y_true)
        return binary_loss,pos_loss,neg_loss,y_pred

    def tra_calc_loss(self,model,data:tuple,weight=None,**kwargs):
        e=kwargs.get('e')
        x,y_true,pad_mask,ignore_index,time=data
        hidden=model(x,pad_mask=pad_mask)
        y_pred, all_preds, prob = self.tra(hidden, time)
        y_true=y_true.view(-1)
        if weight is not None:
            loss=torch.mean(weight*(y_true-y_pred)**2)
        else:
            loss=torch.mean((y_true-y_pred)**2)
        loss = (y_pred - y_true).pow(2).mean()
        L = (all_preds.detach() - y_true[:, None]).pow(2)
        L -= L.min(dim=-1, keepdim=True).values  # normalize & ensure positive input

        if prob is not None:
            P = sinkhorn(-L, epsilon=0.01)  # sample assignment matrix
            lamb = self.lamb * (self.rho**e)
            reg = prob.log().mul(P).sum(dim=-1).mean()
            loss = loss - lamb * reg
        return loss,L.reshape(self.train_batch_size,self.N,self.num_states),y_pred.reshape(self.train_batch_size,-1)
    def mix_calc_loss(self,model,data:tuple,weight=None,**kwargs)->torch.Tensor:
        with torch.no_grad():
            x,y_true,pad_mask,ignore_index,time=data
            y_pred,_=model(x,pad_mask=pad_mask)
            if weight is not None:
                loss=torch.mean(weight*(y_true-y_pred)**2)
            else:
                loss=torch.mean((y_true-y_pred)**2)
            sample_loss=(y_true-y_pred)**2
        _, sorted_indices = torch.sort(sample_loss, dim=1)
        smallest_100_indices = sorted_indices[:, :200]
        largest_100_indices = sorted_indices[:, -200:]
        x[:, largest_100_indices[0], :, :]=0.7*x[:, largest_100_indices[0], :, :]+0.3*x[:, smallest_100_indices[0], :, :]
        y_true[:, largest_100_indices[0]]=0.7*y_true[:, largest_100_indices[0]]+0.3*y_true[:, smallest_100_indices[0]]
        y_pred,_=model(x,pad_mask=pad_mask)
        if weight is not None:
            loss=torch.mean(weight*(y_true-y_pred)**2)
        else:
            loss=torch.mean((y_true-y_pred)**2)

        sample_loss=(y_true-y_pred)**2
#         loss = F.mse_loss(y_pred[~ignore_index], y_true[~ignore_index])
        return loss,sample_loss,y_pred
    def naive_calc_loss(self,model,data:tuple,weight=None,**kwargs)->torch.Tensor:
        x,y_true,pad_mask,ignore_index,time=data
        y_pred,_=model(x,pad_mask=pad_mask)
        if weight is not None:
            loss=torch.mean(weight*(y_true-y_pred)**2)
        else:
            loss=torch.mean((y_true-y_pred)**2)

        sample_loss=(y_true-y_pred)**2
        return loss,sample_loss,y_pred
        
    def multi_calc_loss(self,model,data:tuple,weight=None,**kwargs)->torch.Tensor:
        x,y_true,pad_mask,ignore_index,time=data
        log_true=torch.log(y_true-y_true.min()+1)
        y_true=torch.stack([y_true,log_true],dim=-1)
        y_pred,_=model(x,pad_mask=pad_mask)
        if weight is not None:
            loss=torch.mean(weight*(y_true-y_pred)**2)
        else:
            loss=torch.mean((y_true-y_pred)**2)

        sample_loss=(y_true[...,0]-y_pred[...,0])**2
#         loss = F.mse_loss(y_pred[~ignore_index], y_true[~ignore_index])
        return loss,sample_loss,y_pred[...,0]
    def switch_calc_loss(self,model,data:tuple,weight=None,**kwargs)->torch.Tensor:
        x,y_true,pad_mask,ignore_index,time=data
        y_pred,aux=model(x,pad_mask=pad_mask)
        if weight is not None:
            loss=torch.mean(weight*(y_true-y_pred)**2)+aux
        else:
            loss=torch.mean((y_true-y_pred)**2)+aux

        sample_loss=(y_true-y_pred)**2
#         loss = F.mse_loss(y_pred[~ignore_index], y_true[~ignore_index])
        return loss,sample_loss,y_pred
#     @abstractmethod
    def naive_prep_eval(self, data: tuple) -> tuple:
        # e.g.
        model=self.model_dict['model']
        x, y_true, pad_mask, ignore_index,time = data
        y_pred = model(x, pad_mask=pad_mask)
        if y_pred.ndim==3:
            y_pred=y_pred[...,-1]
        if y_true.ndim==3:
            y_true=y_true[...,-1]
        return y_true, y_pred, ignore_index

    
    def tra_prep_eval(self,data:tuple)->tuple:
        tra,model=self.model_dict['tra'],self.model_dict['model']
        x, y_true, pad_mask, ignore_index,time = data
        hidden = model(x,pad_mask=pad_mask)
        y_pred, all_preds, prob = tra(hidden, time)
        L = (all_preds - y_true.view(-1)[:, None]).pow(2)
        L -= L.min(dim=-1, keepdim=True).values
        return y_pred.view(-1,self.N),y_true.view(-1,self.N),ignore_index,L.reshape(self.valid_batch_size,self.N,self.num_states)
    def set_model(
        self, model_type: type,
        model_config: dict,
        vnet=None,
        tra=None,
        **path_args
    ) -> None:
        """Set model

        Parameters
        ----------
        model_type : type[nn.Module]
            Model type
        state_dict_path : str | None, optional
            Path of model state, by default None
        """
        model_config['L']=self.input_shape['L']
        model_config['C']=self.input_shape['C']
        self.model_dict={}
        if self.meta:
            assert vnet is not None
            self.model_dict['meta']=self._to_device(vnet)
        if self.scheme=='tra':
            assert tra is not None
            self.model_dict['tra']=self._to_device(tra)
        self.model_dict['model']=self._to_device(model_type(model_config))
        for path_name,path in path_args.items():
            name=path_name.split("_")[0]
            state_dict=torch.load(path)
            state_dict={k:v for k,v in zip(self.model_dict[name].state_dict().keys(),state_dict.values())}
            self.model_dict[name].load_state_dict(state_dict)
        if self.verbose:   
            for key,model in self.model_dict.items():
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
        **optimizer_dict_path
    ) -> None:
        """Build pipeline

        Parameters
        ----------
        dataset_type : type[Dataset], optional
            Dataset type, by default MmapDataset
        optimizer_type : type[optim.Optimizer], optional
            Optimizer type, by default optim.AdamW
        warmup_type : type[LRScheduler], optional
            Warmup scheduler type, by default LambdaLR
        anneal_type : type[LRScheduler], optional
            Anneal scheduler type, by default CosineAnnealingLR
        backtester_type : type[Backtester], optional
            Backtester type, by default Backtester
        """
        # dataloaders
        self._init_dataloaders(dataset_type)
        if self.meta:
            self._init_meta_dataloaders(IntraMmapDataset)
        # optimizer
        keys=['model']
        self._init_optimizer(optimizer_type,keys)
        self._init_scheduler(warmup_type, anneal_type,keys)
        for path_name,optimizer in optimizer_dict_path.items():
            name=path_name.split("_")[0]
            self.optimizer_dict[name].load_state_dict(torch.load(optimizer))
        # backtester
        self._init_backtester(backtester_type)
    def all_gather(self,t):
        gathered = [torch.empty_like(t) for _ in range(self.world_size)]
        dist.all_gather(gathered, t)
        t=torch.stack(gathered).transpose(0,1).reshape(-1,*t.shape[1:])
        return t
    def train(self,start=0) -> None:
        """Train model"""
        loss_epoch=[]
#         loss_epoch=[torch.load('loss0.pt').to(self.local_rank),torch.load('loss1.pt').to(self.local_rank),torch.load('loss2.pt').to(self.local_rank)]
#         hist=torch.zeros((len(self.tvt_dates['train'])+1,self.S,self.N,self.num_states))
#         hist=torch.load('model/base_WITRAN_tra_1/base_WITRAN_tra_1_train_0-5_loss.pt')
        for e in range(start,self.epochs):
#             if e==1:break
            # training mode
#             self.dl_train=self.meta_train
            loss_cur=[]
            yp_list=[]
            for model in self.model_dict.values():
                model.train()
            # DDP shuffle
            if self.ddp:
                self.dl_train.sampler.set_epoch(e)
                if self.meta:
                    self.meta_train.sampler.set_epoch(e)
                    self.meta_iter = iter(self.meta_train)
           
            shuffle_index=torch.tensor(list(self.dl_train.sampler.__iter__())).to(self.local_rank)
           
            if self.scheme=='de':
                if e>=2:
                    cur_weight=self.weight_map(cur_loss=loss_epoch[0],init_loss=loss_epoch[-2],num_bins=1000)
                    cur_weight=cur_weight*cur_weight.numel()/cur_weight.sum()
                    cur_weight=cur_weight.reshape(-1,self.stocknum).to(self.local_rank)
                else:
                    cur_weight=None
            
            for b, data in tqdm(enumerate(self.dl_train)):
#                 if b==3:break
                # move tensors to correct device
                batches_per_sub_epoch = len(self.dl_train) // self.sub_epochs
                is_at_sub_e = ((b + 1) % batches_per_sub_epoch) == 0
                sub_e = (b + 1) // batches_per_sub_epoch
                sub_e_passed = e * self.sub_epochs + sub_e
                cur_index=shuffle_index[b*self.train_batch_size:min((b+1)*self.train_batch_size,len(shuffle_index))]
                if self.ddp:
                    data = [i.to(self.local_rank) for i in data]
                else:
                    data = [i.to(self.device) for i in data]
                if self.vae:
#                     optimizer=self.init_optimizer(model,optim.AdamW)
                    optimizer=self.optimizer_dict['model']
                    for _ in range(2):
                        optimizer.zero_grad()
                        model=self.model_dict['model']
                        vae_loss=self.vae_loss(model,data)
                        vae_loss.backward()
                        optimizer.step()
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
                    model,vnet,optimizer,vnet_optimizer=self.model_dict['model'],self.model_dict['meta'],self.optimizer_dict['model'],self.optimizer_dict['meta']
                    temp_model=copy.deepcopy(model).to(self.local_rank)
                    temp_optimizer=self.init_optimizer(temp_model,optim.AdamW)
                    temp_optimizer.load_state_dict(optimizer.state_dict())
                    _,sample_loss,_=self.naive_calc_loss(temp_model,data,weight=None)
                    sample_loss=sample_loss.unsqueeze(-1)
                    
                    v_weight=vnet(sample_loss,data[1].unsqueeze(-1)).squeeze(-1)
                    v_weight=v_weight*v_weight.numel()/v_weight.sum()
                    loss=torch.mean(sample_loss*v_weight)
                    temp_optimizer.zero_grad()
                    loss.backward()
                    temp_optimizer.step()              
                    meta_loss,_,_=self.naive_calc_loss(temp_model,data)
                    vnet_optimizer.zero_grad()
                    meta_loss.backward()
                    vnet_optimizer.step()
                    del temp_model,temp_optimizer
                    optimizer.zero_grad()
                    _,sample_loss ,_= self.loss_func(model,data)
                    sample_loss=sample_loss.unsqueeze(-1)
                    with torch.no_grad():
                        v_weight=vnet(sample_loss,data[1].unsqueeze(-1)).squeeze(-1)
                        v_weight=v_weight*v_weight.numel()/v_weight.sum()
#                     print(v_weight.shape)
                    loss,sample_loss,y_pred=self.naive_calc_loss(model,data,weight=v_weight)
                    loss.backward()
                    optimizer.step()
                    loss_cur.append(sample_loss.detach())
                
                else:
            
#                     cur_index=[i//self.S for i in cur_index] 
#                     hist_loss=hist[cur_index].transpose(1,2).reshape(-1,self.S,self.num_states) (tra)

                            
                    model,optimizer=self.model_dict['model'],self.optimizer_dict['model']
                    weight=cur_weight[cur_index] if self.scheme=='de' is not None and e>=2 else None
#                     weight=torch.load('mse_valid_weight.pt').to(self.local_rank)
                    optimizer.zero_grad()
                    loss,sample_loss,y_pred=self.loss_func(model,data=data,weight=weight)
#                     print(loss.shape,sample_loss.shape)
#                     weight=self.pace_weight(sample_loss,a=torch.quantile(sample_loss,0.9),b=torch.quantile(sample_loss,0.1)).to(self.local_rank)
#                     weight=weight*weight.numel()/torch.sum(weight)
#                     loss=torch.mean(weight*sample_loss)
                    loss_cur.append(sample_loss.detach())
                    loss.backward()
                    optimizer.step()

#                 naive
#                 elif self.scheme=='step':
#                     y_pred=torch.zeros(data[1].shape).to(self.local_rank)
#                     class_loss,pos_index=self.step_calc_loss(data,signal='class')
#                     self.optimizer.zero_grad()
#                     class_loss.backward()
#                     self.optimizer.step()
#                     pos_loss,pos_pred=self.step_calc_loss(data,signal='pos')
#                     self.optimizer.zero_grad()
#                     pos_loss.backward()
#                     self.optimizer.step()
#                     neg_loss,neg_pred=self.step_calc_loss(data,signal='neg')
#                     self.optimizer.zero_grad()
#                     neg_loss.backward()
#                     self.optimizer.step()
#                     y_pred[pos_index]=pos_pred.clone().detach()
#                     y_pred[~pos_index]=neg_pred.clone().detach()
                torch.cuda.empty_cache()
#                 for name, param in self.model.named_parameters():
#                     if param.grad is not None:
#                         grad_norm = param.grad.norm().item()
#                         self.writer.add_scalar(f'GradNorm/{name}', grad_norm, b)
#                         if b%1000==0:
#                             print(grad_norm)
                # check sub epoch

                yp_list.append(y_pred.detach())
                
                # scheduler update
#                 is_at_sub_e=True
                if sub_e_passed < self.warmup_end:
                    for warmup in self.warmup_group:
                        warmup.step()
                if is_at_sub_e and sub_e_passed >= self.anneal_start:
                    for anneal in self.anneal_group:
                        anneal.step()
#                 # save model
                if not self.local_rank and is_at_sub_e:

                    for key,model in self.model_dict.items():
                        torch.save(
                        model.state_dict(),
                        self.dir.m_exp + f"{self.model_name}_{key}_{e}-{sub_e}.pt",
                    )
                    for key,optimizer in self.optimizer_dict.items():
                        torch.save(
                        optimizer.state_dict(),
                        self.dir.m_exp + f"{self.model_name}_{key}_{e}-{sub_e}_optimizer.pt",
                    )
#                 validate model
                if self.enable_valid and is_at_sub_e:
                    self.evaluate("valid", True, f"{e}-{sub_e}",True,e,sub_e)
                    self.evaluate("test",True, f"{e}-{sub_e}",True,e,sub_e)
                    for model in self.model_dict.values():
                        model.train()
            
            loss_cur=torch.cat(loss_cur).to(self.local_rank)
            yp=torch.cat(yp_list)
            del yp_list    
            mode='train'
            if self.ddp:
                loss_cur=self.all_gather(loss_cur)
                yp=self._dist_gather(yp,mode)
                shuffle_index=self.all_gather(shuffle_index)
            recovery_indices = torch.argsort(shuffle_index).unsqueeze(-1).expand(loss_cur.shape)
            loss_cur=torch.gather(input=loss_cur,dim=0,index=recovery_indices).contiguous()
            loss_epoch.append(loss_cur)
#             print(loss_cur.shape)        
#             print(len(loss_epoch))
#                     if self.scheme=='tra' and not self.local_rank:
#                         shuffle_index=shuffle_index.unsqueeze(-1).unsqueeze(-1).expand(loss_cur.shape).contiguous()
#                         loss_cur=torch.gather(input=loss_cur,dim=0,index=shuffle_index).contiguous()
#                 shuffle_index=self._dist_gather(shuffle_index,mode)
#                 yp=self._dist_gather(yp,mode)
            
#             if self.scheme=='tra'and not self.local_rank:
#                 temp=torch.zeros(self.index_base+1,self.S,self.N,self.num_states).to(self.local_rank)
#                 hist=loss_cur.reshape(len(self.tvt_dates[mode])-self.index_base,self.S,self.N,self.num_states)
#                 hist=torch.cat((temp,hist),dim=0).contiguous()
#                 dist.broadcast(hist,src=0)

#             print(self.local_rank,hist.shape)
            if not self.local_rank:
                recover_indices= torch.argsort(shuffle_index).unsqueeze(-1).expand(yp.shape)
                yp=torch.gather(input=yp,dim=0,index=recovery_indices)
                a,b=yp.shape
                groupnum=int(np.ceil(self.N/self.stocknum))
                yp=yp.reshape(a//groupnum,b*groupnum)
                stockslice=list(range(self.N//self.stocknum*self.stocknum))
                stockslice.extend(list(range(b*groupnum-self.N%self.stocknum,b*groupnum)))
                yp=yp[:,stockslice]
                torch.save(yp,self.dir.r_exp + f'{self.model_name}_{mode}_{e}_pred.pt')

        self.writer.close()

    @torch.no_grad()
    def evaluate(self, mode: str, save: bool = False, save_suffix: str = "",save_res=False,e=0,sub_e=0) -> None:
#         print(e,sub_e)
        """Evaluate model

        Parameters
        ----------
        mode : str
            Operation mode, choose from ["valid", "test"]
        save : bool, optional
            If save results, by default False
        save_suffix : str, optional
            Evaluating results save suffix, by default ""
        """
        groupnum=int(np.ceil(self.N/self.stocknum))
        loss_cur=[]
        if mode == "valid":
            dl_eval = self.dl_valid
        elif mode == "test":
            dl_eval = self.dl_test
        elif mode=='train':
            self.dl_train=self._gen_dataloader(self.datasets['train'],mode='test')
            dl_eval = self.dl_train
        else:
            raise ValueError(f"{mode} is not supported")

        for model in self.model_dict.values():
            model.eval()
        # synchronize processes
        if self.ddp:
            dist.barrier()
#         shuffle_index=torch.tensor(list(dl_eval.sampler.__iter__())).to(self.local_rank)
        # evaluate
       
        yt_list, yp_list, ii_list = [], [], []
#         hist=torch.zeros((len(self.tvt_dates[mode])+1,self.S,self.N,self.num_states)).to(self.local_rank)

        for index,data in tqdm(enumerate(dl_eval)):
#             cur_index=shuffle_index[index*self.valid_batch_size:min((index+1)*self.valid_batch_size,len(shuffle_index))]
            # move tensors to correct device

            if self.ddp:
                data = [i.to(self.local_rank) for i in data]
            else:
                data = [i.to(self.device) for i in data]
            # process batch
            if self.scheme=='stack':
                yt, yp, ii = self.stack_prep_eval(data)
            elif self.scheme=='tra':
#                 temp=[i//self.S for i in cur_index]
#                 hist_loss=hist[temp].transpose(1,2).reshape(-1,self.S,self.num_states)
                yp,yt,ii,sample_loss=self.tra_prep_eval(data)
 
#                 cur_index=self._dist_gather(cur_index,mode)
#                 sample_loss=self._dist_gather(sample_loss,mode)
#                 date=[i//self.S+self.index_base+1 for i in cur_index]
#                 time=[i%self.S for i in cur_index]
#                 hist[date,time]=sample_loss
#                 dist.broadcast(hist,src=0)

            else:
                yt,yp,ii=self.naive_prep_eval(data)
#                 print(yp.shape,yt.shape,ii.shape)
            yt_list.append(yt)
            yp_list.append(yp)
            ii_list.append(ii)
        # concatenate across batches
        yt = torch.cat(yt_list)
        yp = torch.cat(yp_list)
        ii = torch.cat(ii_list)
       
#         print(self.local_rank,yp.shape) s
        if self.ddp:
            yt = self._dist_gather(yt, mode)
            yp = self._dist_gather(yp, mode)
            ii = self._dist_gather(ii, mode)
#         if not self.local_rank:
#             torch.save(hist,self.dir.m_exp + f'{self.model_name}_{mode}_{e}-{sub_e}_loss.pt')
#         if self.sequence:
#             yt=yt.view(-1,self.N)
#             yp=yp.view(-1,self.N)
#             ii=ii.view(-1,self.N)
       
        if save and not self.local_rank:
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
        
