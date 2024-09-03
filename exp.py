import sys
sys.path.append('/dfs/data/package')
from stm import EncoderPipeline
from stm.data import Preprocessor
# from stm.BaseEncoder.model import BaseEncoder
# from stm.lstm_transformer.lstm_transformer import lstm_transformer
# from stm.utils import Directory
# from stm.MLPTSMixer.tsMixer import TSMixerModel
# from stm.MTSMixer.MTSMixer import *
# from stm.MTSMixer.MTSMixer import MTSMixer
# from stm.DLinear.DLinear import DLinear
# from stm.MogLSTM.lstm import MogLSTM
# from stm.WITRAN.WITRAN import WITRAN
# from stm.VNet.VNet import VNet
# from stm.SFM_WITRAN.SFM_WITRAN import SFM_WITRAN
# from stm.fedformer.fedformer import fedformer
# from stm.TRA.TRA import TRA
# from stm.stack_witran.stack_witran import stack_witran
import torch
import argparse
import yaml
from tensorboardX import SummaryWriter
import os

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
parser = argparse.ArgumentParser(description='test')

dir_main = "/dfs/data/"
dir = Directory(dir_main)
parser.add_argument('--model_config_name', type=str)
parser.add_argument('--model_type', type=str)
parser.add_argument('--model_name', type=str)
args = parser.parse_args()

with open(f'stm/{args.model_type}/configs.yaml') as f:
    model_configs = yaml.safe_load(f)
args.model = eval(args.model_type)
print(args)
configs_filename = "configs.yaml"

with open(f'config/configs.yaml') as file:
    configs = yaml.safe_load(file)
configs['model']['name'] = args.model_name
configs['model']['writer'] = f'/dfs/data/log/{args.model_name}'
# print(config)
with open(f'config/configs.yaml', 'w') as file:
    yaml.safe_dump(configs, file)
vnet = VNet(1,3000,1)
model_configs['L'] = 64
model_configs['C'] = 170
model = args.model(model_configs)
x = torch.rand(1, 100, 64, 170)
if configs['scheme']['type'] == 'tra':
    input_size = model(x, pad_mask=None)[0].shape[-1]
    tra = TRA(input_size=input_size)

## Columns
# ticker_col = "symbol"
# x_cols = [f"f{i}" for i in range(1, 171)]
# y_col = "yhat_raw_LnRet_v2v_1d_15m_v1"
# ticker_col_x = "symbol"
# ticker_col_y = "skey"

## Preprocessing
# pre = Preprocessor(dir, configs_filename)
# pre.scan(ticker_col)
# pre.clean(x_cols, y_col, ticker_col_x, ticker_col_y)

## Pipeline
pipe = EncoderPipeline(dir, configs_filename, True)
pipe.set_model(model_type=args.model, vnet=vnet, model_config=model_configs)
pipe.build()
pipe.train(start=0)
# pipe.set_model(model_type=args.model, vnet=vnet, tra=tra, model_config=model_configs,
#               state_dict_path='model/base_WITRAN_tra_timeembed_1/base_WITRAN_tra_timeembed_1_0-5.pt',
#               tra_state_dict_path='model/base_WITRAN_tra_timeembed_1/base_WITRAN_tra_timeembed_1_0-5.pt')
# pipe.build(optimizer_dict_path='model/base_WITRAN_tra_timeembed_1/base_WITRAN_tra_timeembed_1_0-5_optimizer.pt')
# pipe.train(start=1)
# pipe.evaluate("test", True)
# pipe.evaluate("valid", True)


