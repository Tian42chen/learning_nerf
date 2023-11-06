import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from lib.networks.encoding import get_encoder
from lib.config import cfg

class Network(nn.Module):
    def __init__(self,):
        super(Network, self).__init__()
        net_cfg = cfg.network
        self.uv_encoder, input_ch = get_encoder(net_cfg.uv_encoder)
        self.chunk_size = cfg.task_arg.chunk_size
        raise NotImplementedError
    
    def render(self, uv, batch):
        uv_encoding = self.uv_encoder(uv)
        raise NotImplementedError
    
    def batchify(self, uv, batch):
        raise NotImplementedError
    
    def forward(self, batch):
        raise NotImplementedError
              