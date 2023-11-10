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

    def forward(self, pts, viewdir, dist):
        x = self.xyz_encoder(pts)
        for l in self.xyz_linears:
            x = l(x)
            x = F.relu(x)
        rawalpha = self.alpha_linear(x)
        feature = self.feature_linear(x)

        dir_encoding = self.dir_encoder(viewdir)
        x = torch.cat([feature, dir_encoding], dim=-1)
        for l in self.dir_linears:
            x = l(x)
            x = F.relu(x)
        rgb = self.rgb_linear(x)

        alpha = 1. - torch.exp(-F.relu(rawalpha)*dist)

        return alpha, rgb

class Network(nn.Module):
    def __init__(self,):
        super(Network, self).__init__()
        net_cfg = cfg.network

        self.chunk_size = cfg.task_arg.chunk_size
        self.white_bkgd = cfg.task_arg.white_bkgd
        self.perturb = cfg.task_arg.perturb
        self.N_samples = cfg.task_arg.cascade_samples[0]
        self.N_importance = None

        self.coarse_net = NeRF(net_cfg)
        if len(cfg.task_arg.cascade_samples)>1:
            self.N_importance = cfg.task_arg.cascade_samples[1]
            self.fine_net = NeRF(net_cfg)

    def render(self, ray_o, ray_d, near, far, batch):
        t_vals = torch.linspace(0., 1., steps=self.N_samples, device=near.device, dtype=near.dtype)
        z_vals = near * (1. - t_vals) + far * t_vals

        pts, viewdir, dists = get_5D_coords(ray_o, ray_d, z_vals, self.perturb)

        if cfg.debug:
            # save pts as npy
            save_debug(pts, 'pts')
            save_debug(viewdir, 'viewdir')

        N, S, C = pts.shape

        pts = pts.reshape(-1, C)
        viewdir = viewdir.reshape(-1, C)
        dists = dists.reshape(-1,1)

        alpha, rgb = self.coarse_net(pts, viewdir, dists)

        rgb = rgb.reshape(N, S, -1)
        alpha = alpha.reshape(N, S)

        weights, rgb_map, acc_map = volume_rendering(rgb, alpha, bg_brightness=self.white_bkgd)

        coarse_ret = {'rgb': rgb_map, 'weights': weights, 'acc': acc_map}

        fine_ret = None
        if self.N_importance is not None:
            # importance sampling
            z_vals_mid = .5 * dists.reshape(N,S)[..., :-1]
            z_samples = sample_pdf(z_vals_mid, weights[...,1:-1], self.N_importance, perturb=self.perturb)
            fine_z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)
            fine_pts, fine_viewdir, fine_dists = get_5D_coords(ray_o, ray_d, fine_z_vals, self.perturb)

            fine_N, fine_S, fine_C = fine_pts.shape

            fine_pts = fine_pts.reshape(-1, fine_C)
            fine_viewdir = fine_viewdir.reshape(-1, fine_C)
            fine_dists = fine_dists.reshape(-1,1)

            fine_alpha, fine_rgb = self.fine_net(fine_pts, fine_viewdir, fine_dists)

            fine_rgb = fine_rgb.reshape(fine_N, fine_S, -1)
            fine_alpha = fine_alpha.reshape(fine_N, fine_S)

            fine_weights, fine_rgb_map, fine_acc_map = volume_rendering(fine_rgb, fine_alpha, bg_brightness=self.white_bkgd, bg_image=None)

            fine_ret = {'rgb': fine_rgb_map, 'weights': fine_weights, 'acc': fine_acc_map}

        if fine_ret is None: ret = coarse_ret
        else:
            ret = {}
            for k in coarse_ret:
                ret[k+'_coarse'] = coarse_ret[k]
            for k in fine_ret:
                ret[k] = fine_ret[k]
                
        if cfg.debug:
            for k in ret:
                if (torch.isnan(ret[k]).any() or torch.isinf(ret[k]).any()):
                    print(f"! [Numerical Error] {k} contains nan or inf.")

        return ret
    
    def batchify(self, ray_o, ray_d, near, far, batch):
        all_ret = {}
        for i in range(0, ray_o.shape[0], self.chunk_size):
            ret = self.render(ray_o[i:i + self.chunk_size], ray_d[i:i + self.chunk_size], near[i:i + self.chunk_size], far[i:i + self.chunk_size], batch)
            for k in ret:
                if k not in all_ret:
                    all_ret[k] = []
                all_ret[k].append(ret[k])
        all_ret = {k: torch.cat(all_ret[k], dim=0) for k in all_ret}
        return all_ret
    
    def forward(self, batch):
        B, N_rays, C = batch['rays_o'].shape
        ret = self.batchify(batch['rays_o'].reshape(-1, C), batch['rays_d'].reshape(-1, C), batch['near'].reshape(-1,1), batch['far'].reshape(-1,1),  batch)
        return {k:ret[k].reshape(B, N_rays, -1) for k in ret}