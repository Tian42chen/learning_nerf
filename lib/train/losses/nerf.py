import torch
import torch.nn as nn
from lib.utils import net_utils
from lib.config import cfg

class NetworkWrapper(nn.Module):
    def __init__(self, net):
        super(NetworkWrapper, self).__init__()
        self.net=net

    def forward(self,batch):
        output=self.net(batch)
        raise NotImplementedError
        scalar_stats={}
        loss=0
        scalar_stats.update({'loss':loss})
        image_stats={}
        return output,loss,scalar_stats,image_stats