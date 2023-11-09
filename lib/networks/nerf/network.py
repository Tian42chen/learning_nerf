import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from lib.networks.encoding import get_encoder
from lib.utils.if_nerf_utils import volume_rendering
from lib.config import cfg

class Network(nn.Module):
    def __init__(self,):
        super(Network, self).__init__()
        net_cfg = cfg.network
        self.xyz_encoder, input_xyz_ch = get_encoder(net_cfg.xyz_encoder)
        self.dir_encoder, input_dir_ch = get_encoder(net_cfg.dir_encoder)

        W, D, V_D  = net_cfg.nerf.W, net_cfg.nerf.D, net_cfg.nerf.V_D
        self.chunk_size = cfg.task_arg.chunk_size
        self.N_samples = cfg.task_arg.cascade_samples[0]

        self.xyz_linears = nn.ModuleList(
            [nn.Linear(input_xyz_ch, W)] + [nn.Linear(W, W) for i in range(D-1)])
        self.dir_linears = nn.ModuleList([nn.Linear(input_dir_ch + W, W//2)])

        self.feature_linear = nn.Linear(W, W)
        self.alpha_linear = nn.Linear(W, 1)
        self.rgb_linear = nn.Linear(W//2, 3)
    
    def get_5D_coords(self, ray_o, ray_d, near, far):
        t_vals = torch.linspace(0., 1., steps=self.N_samples, device=near.device, dtype=near.dtype)
        z_vals = near * (1. - t_vals) + far * t_vals

        if cfg.perturb > 0.:# and self.net.training:
            # get intervals between samples
            mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
            upper = torch.cat([mids, z_vals[..., -1:]], -1)
            lower = torch.cat([z_vals[..., :1], mids], -1)
            # stratified samples in those intervals
            t_rand = torch.rand(z_vals.shape, device=upper.device, dtype=upper.dtype)
            z_vals = lower + (upper - lower) * t_rand

        pts = ray_o[:,None, :] + ray_d[:,None, :]  * z_vals [..., None] # (N_rays, N_samples, 3)
        viewdir = ray_d[:,None, :].expand(pts.shape) # (N_rays, N_samples, 3)

        # calculate dists for the opacity computation
        dists = z_vals[..., 1:] - z_vals[..., :-1]
        dists = torch.cat([dists, dists[..., -1:]], dim=-1) # (N_rays, N_samples)
        # dists = dists.view(n_batch * n_pixel * n_sample)
        return pts, viewdir, dists
    
    def get_density_color(self, pts, viewdir, dist):
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

        alpha = 1 - torch.exp(-F.relu(rawalpha)*dist)

        return alpha, rgb

    def render(self, ray_o, ray_d, near, far, batch):
        pts, viewdir, dists = self.get_5D_coords(ray_o, ray_d, near, far)

        N, S, C = pts.shape

        pts = pts.reshape(-1, C)
        viewdir = viewdir.reshape(-1, C)
        dists = dists.reshape(-1,1)

        alpha, rgb = self.get_density_color(pts, viewdir, dists)

        rgb = rgb.reshape(N, S, -1)
        alpha = alpha.reshape(N, S)

        weights, rgb_map, acc_map = volume_rendering(rgb, alpha, bg_brightness=None, bg_image=None)

        ret = {'rgb': rgb_map, 'weights': weights, 'acc': acc_map}

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