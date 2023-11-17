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

        W, D, V_D  = net_cfg.nerf.W, net_cfg.nerf.D, net_cfg.nerf.V_D
        input_xyz_ch, input_dir_ch = net_cfg.input_xyz_ch, net_cfg.input_dir_ch

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
        nn.init.constant_(self.alpha_linear.bias, 0)
        if cfg.debug:
            print("alpha_linear.weight: ", self.alpha_linear.weight.mean())
            print("alpha_linear.weight: ", self.alpha_linear.weight.max())
            print("alpha_linear.weight: ", self.alpha_linear.weight.min())
            print("alpha_linear.bias: ", self.alpha_linear.bias.mean())
            print("alpha_linear.bias: ", self.alpha_linear.bias.max())
            print("alpha_linear.bias: ", self.alpha_linear.bias.min())

    def forward(self, pts, viewdir):
        x = pts
        for l in self.xyz_linears:
            x = l(x)
            x = F.relu(x)
        # debug_x = x.clone()
        rawalpha = self.alpha_linear(x)
        feature = self.feature_linear(x)

        x = torch.cat([feature, viewdir], dim=-1)
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

        self.chunk_size = cfg.task_arg.chunk_size
        self.white_bkgd = cfg.task_arg.white_bkgd
        self.N_samples = cfg.task_arg.cascade_samples[0]
        self.N_importance = None

        self.xyz_encoder, input_xyz_ch = get_encoder(net_cfg.xyz_encoder)
        self.dir_encoder, input_dir_ch = get_encoder(net_cfg.dir_encoder)

        net_cfg.input_xyz_ch = input_xyz_ch
        net_cfg.input_dir_ch = input_dir_ch

        self.coarse_net = NeRF(net_cfg)
        if len(cfg.task_arg.cascade_samples)>1:
            self.N_importance = cfg.task_arg.cascade_samples[1]
            self.fine_net = NeRF(net_cfg)

        self.i=0

    def render(self, rays_o, rays_d, batch):
        if 'perturb' in batch['meta']: perturb = 1
        else: perturb =0

        near, far = batch['near']*torch.ones_like(rays_o[..., :1]), batch['far']*torch.ones_like(rays_o[..., :1])

        t_vals = torch.linspace(0., 1., steps=self.N_samples, device=near.device, dtype=near.dtype)
        z_vals = near * (1. - t_vals) + far * t_vals

        pts, viewdir, dists = get_5D_coords(rays_o, rays_d, z_vals, perturb)

        if cfg.debug:
            # save pts as npy
            # print(f"network{self.i}: ",ray_o[0])
            save_debug(rays_o[0], f'ray_o{self.i}')
            save_debug(pts, f'pts{self.i}')
            save_debug(viewdir, f'viewdir{self.i}')
            self.i+=1

        N, S, C = pts.shape

        pts = pts.reshape(-1, C)
        viewdir = viewdir.reshape(-1, C)

        xyz_encoding = self.xyz_encoder(pts)
        dir_encoding = self.dir_encoder(viewdir)

        ret = self.batchify(xyz_encoding, dir_encoding, self.coarse_net)

        rawalpha = ret['rawalpha'].reshape(N, S)
        rgb = ret['rgb'].reshape(N, S, -1)

        alpha = 1. - torch.exp(-F.relu(rawalpha)*dists)

        if cfg.debug and alpha.max() <= 0.:
            print("rawalpha: ", rawalpha.mean())
            print("rawalpha: ", rawalpha.max())
            print("alpha: ", alpha.mean())
            print("alpha: ", alpha.max())
            raise Exception("alpha < 0")

        weights, rgb_map, acc_map = volume_rendering(rgb, alpha, bg_brightness=self.white_bkgd)

        coarse_ret = {'rgb': rgb_map, 'weights': weights, 'acc': acc_map}

        fine_ret = None
        if self.N_importance is not None:
            # importance sampling
            z_vals_mid = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
            z_samples = sample_pdf(z_vals_mid, weights[...,1:-1], self.N_importance, perturb=perturb)
            fine_z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)
            fine_pts, fine_viewdir, fine_dists = get_5D_coords(rays_o, rays_d, fine_z_vals)

            fine_N, fine_S, fine_C = fine_pts.shape

            fine_pts = fine_pts.reshape(-1, fine_C)
            fine_viewdir = fine_viewdir.reshape(-1, fine_C)

            fine_xyz_encoding = self.xyz_encoder(fine_pts)
            fine_dir_encoding = self.dir_encoder(fine_viewdir)

            fine_ret = self.batchify(fine_xyz_encoding, fine_dir_encoding, self.fine_net)

            fine_rawalpha = fine_ret['rawalpha'].reshape(fine_N, fine_S)
            fine_rgb = fine_ret['rgb'].reshape(fine_N, fine_S, -1)

            fine_alpha = 1. - torch.exp(-F.relu(fine_rawalpha)*fine_dists)

            fine_weights, fine_rgb_map, fine_acc_map = volume_rendering(fine_rgb, fine_alpha, bg_brightness=self.white_bkgd)

            fine_ret = {'rgb': fine_rgb_map, 'weights': fine_weights, 'acc': fine_acc_map}

        if fine_ret is None: ret = coarse_ret
        else:
            ret = fine_ret
            for k in coarse_ret:
                ret[k+'_coarse'] = coarse_ret[k]
                
        if cfg.debug:
            # print("z_vals: ", z_vals.shape, z_vals[0])
            # print("z_vals_mid: ", z_vals_mid.shape, z_vals_mid[0])
            # print("fine_z_vals: ", fine_z_vals.shape, fine_z_vals[0])
            save_debug(rgb, f'rgb')
            save_debug(alpha, f'alpha')
            for k in ret:
                if (torch.isnan(ret[k]).any() or torch.isinf(ret[k]).any()):
                    print(f"! [Numerical Error] {k} contains nan or inf.")

        return ret
    
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
    
    def forward(self, batch):
        B, N_rays, C = batch['rays_o'].shape
        ret = self.render(batch['rays_o'].reshape(-1, C), batch['rays_d'].reshape(-1, C), batch)
        return {k:ret[k].reshape(B, N_rays, -1) for k in ret}