import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from lib.networks.encoding import get_encoder
from lib.utils.if_nerf_utils import volume_rendering, get_5D_coords, sample_pdf
from lib.utils.debug_utils import save_debug
from lib.config import cfg

class Renderer():
    def __init__(self, net) -> None:
        self.net = net

        self.chunk_size = cfg.task_arg.chunk_size
        self.white_bkgd = cfg.task_arg.white_bkgd
        self.N_samples = cfg.task_arg.cascade_samples[0]
        self.N_importance = None

        if len(cfg.task_arg.cascade_samples)>1:
            self.N_importance = cfg.task_arg.cascade_samples[1]

    def render_rays(self, rays_o, rays_d, batch):
        if 'perturb' in batch['meta']: perturb = 1
        else: perturb =0

        near, far = batch['near']*torch.ones_like(rays_o[..., :1]), batch['far']*torch.ones_like(rays_o[..., :1])

        t_vals = torch.linspace(0., 1., steps=self.N_samples, device=near.device, dtype=near.dtype)
        z_vals = near * (1. - t_vals) + far * t_vals

        pts, viewdir, dists = get_5D_coords(rays_o, rays_d, z_vals, perturb)

        N, S, C = pts.shape

        pts = pts.reshape(-1, C)
        viewdir = viewdir.reshape(-1, C)

        ret = self.net(pts, viewdir)

        rawalpha = ret['rawalpha'].reshape(N, S)
        rgb = ret['rgb'].reshape(N, S, -1)

        alpha = 1. - torch.exp(-F.relu(rawalpha)*dists)

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

            fine_ret = self.net(fine_pts, fine_viewdir, 'fine')

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

        return ret


    def batchify(self, rays_o, rays_d, batch):
        all_ret = {}
        for i in range(0, rays_o.shape[0], self.chunk_size):
            ret = self.render_rays(rays_o[i:i+self.chunk_size], rays_d[i:i+self.chunk_size], batch)
            for k in ret:
                if k not in all_ret:
                    all_ret[k] = []
                all_ret[k].append(ret[k])
        all_ret = {k: torch.cat(all_ret[k], dim=0) for k in all_ret}
        return all_ret

    def render(self, batch):
        B, N_rays, C = batch['rays_o'].shape
        ret = self.batchify(batch['rays_o'].reshape(-1, C), batch['rays_d'].reshape(-1, C), batch)
        return {k:ret[k].reshape(B, N_rays, -1) for k in ret}