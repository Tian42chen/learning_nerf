import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from lib.networks.encoding import get_encoder
from lib.utils.if_nerf_utils import volume_rendering, get_5D_coords, sample_pdf
from lib.utils.debug_utils import save_debug
from lib.config import cfg

class NeRF(nn.Module):
    def __init__(self, net_cfg) -> None:
        super(NeRF, self).__init__()
        self.xyz_encoder, input_xyz_ch = get_encoder(net_cfg.xyz_encoder)
        self.dir_encoder, input_dir_ch = get_encoder(net_cfg.dir_encoder)

        W, D, V_D  = net_cfg.nerf.W, net_cfg.nerf.D, net_cfg.nerf.V_D

        self.xyz_linears = nn.ModuleList(
            [nn.Linear(input_xyz_ch, W)] + [nn.Linear(W, W) for i in range(D-1)])
        self.dir_linears = nn.ModuleList(
            [nn.Linear(input_dir_ch + W, W//2)]  + [nn.Linear(W//2, W//2) for i in range(V_D-1)])

        self.feature_linear = nn.Linear(W, W)
        self.alpha_linear = nn.Linear(W, 1)
        self.rgb_linear = nn.Linear(W//2, 3)
        # F.leaky_relu(self.rgb_linear.weight, 0.1, inplace=True)
        # NOTE 若 rawalpha 初始得到负数, 则经过 relu 后梯度永远不可能传递.
        # nn.init.uniform_(self.alpha_linear.weight, -.5/np.sqrt(W), 1./np.sqrt(W))
        # nn.init.uniform_(self.alpha_linear.bias, -.5/np.sqrt(W), 1./np.sqrt(W))
        # nn.init.constant_(self.alpha_linear.bias, 0)
        if cfg.debug:
            print("alpha_linear.weight: ", self.alpha_linear.weight.mean())
            print("alpha_linear.weight: ", self.alpha_linear.weight.max())
            print("alpha_linear.weight: ", self.alpha_linear.weight.min())
            print("alpha_linear.bias: ", self.alpha_linear.bias.mean())
            print("alpha_linear.bias: ", self.alpha_linear.bias.max())
            print("alpha_linear.bias: ", self.alpha_linear.bias.min())

    def forward(self, pts, viewdir):
        x = self.xyz_encoder(pts)
        for l in self.xyz_linears:
            x = l(x)
            x = F.relu(x)
        # debug_x = x.clone()
        rawalpha = self.alpha_linear(x)
        feature = self.feature_linear(x)

        dir_encoding = self.dir_encoder(viewdir)
        x = torch.cat([feature, dir_encoding], dim=-1)
        for l in self.dir_linears:
            x = l(x)
            x = F.relu(x)
        rgb = self.rgb_linear(x)
        rgb = torch.sigmoid(rgb)

        # print("pts: ",pts.mean())
        # print("dist: ",dist.mean())
        # if cfg.debug and rawalpha.max() <0.:
        #     # print("debug_x: ", debug_x[0])
        #     # print("weights: ", self.alpha_linear.weight)
        #     # print("debug_x: ", debug_x[0].max())
        #     # print("debug_x: ", debug_x[0].mean())
        #     # print("debug_x: ", debug_x[0].min())
        #     print("rawalpha: ",rawalpha[0].mean())
        #     print("rawalpha: ",rawalpha[0].max())
        #     # print("alpha: ",alpha[0].mean())
        #     raise Exception("rawalpha < 0")

        return {'rgb': rgb, 'rawalpha': rawalpha}

class Network(nn.Module):
    def __init__(self,):
        super(Network, self).__init__()
        net_cfg = cfg.network

        self.chunk_size = cfg.task_arg.chunk_size*16

        self.coarse_net = NeRF(net_cfg)
        if len(cfg.task_arg.cascade_samples)>1:
            self.fine_net = NeRF(net_cfg)

    def batchify(self, pts, viewdir, fn):
        all_ret = {}
        for i in range(0, pts.shape[0], self.chunk_size):
            ret = (fn)(pts[i:i + self.chunk_size], viewdir[i:i + self.chunk_size])
            for k in ret:
                if k not in all_ret:
                    all_ret[k] = []
                all_ret[k].append(ret[k])
        all_ret = {k: torch.cat(all_ret[k], dim=0) for k in all_ret}
        return all_ret
    
    def forward(self, pts, viewdir, model=''):
        fn = self.fine_net if model == 'fine' else self.coarse_net
        ret = self.batchify(pts, viewdir, fn)
        return ret