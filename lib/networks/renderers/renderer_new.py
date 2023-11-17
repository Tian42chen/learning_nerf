import torch
import torch.nn.functional as F
from lib.config import cfg
import numpy as np


class Renderer:
    def __init__(self, net):
        self.net = net


    def render_rays(self, ray_batch, net_c=None, pytest=False):
        N_rays = ray_batch.shape[0]
        rays_o, rays_d = ray_batch[:, 0:3], ray_batch[:, 3:6]  # [N_rays, 3] each
        viewdirs = ray_batch[:, -3:] if ray_batch.shape[-1] > 8 else None
        bounds = torch.reshape(ray_batch[..., [6, 8]], [-1, 1, 2])
        # Fix some bug here, near and far with the shape of (N_rays, 2)
        # bounds = torch.reshape(ray_batch[..., 6:8], [-1, 1, 2])
        near, far = bounds[..., 0], bounds[..., 1]  # [-1,1]

        t_vals = torch.linspace(0., 1., steps=64).to(near)
        z_vals = near * (1. - t_vals) + far * (t_vals)
        z_vals = z_vals.expand([N_rays, 64])

        if True and self.net.training:
            # get intervals between samples
            mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
            upper = torch.cat([mids, z_vals[..., -1:]], -1)
            lower = torch.cat([z_vals[..., :1], mids], -1)
            # stratified samples in those intervals
            t_rand = torch.rand(z_vals.shape).to(upper)

            # Pytest, overwrite u with numpy's fixed random numbers
            if pytest:
                np.random.seed(0)
                t_rand = np.random.rand(*list(z_vals.shape))
                t_rand = torch.Tensor(t_rand)

            z_vals = lower + (upper - lower) * t_rand

        pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[
            ..., :, None]  # [N_rays, N_samples, 3]
        
        N, S, C = pts.shape

        input_pts = pts.reshape(-1, C)
        input_viewdirs = viewdirs[:, None].expand(pts.shape).reshape(-1, 3)

        ret = self.net(input_pts, input_viewdirs)

        outputs_flat = torch.cat([ret['rgb'], ret['rawalpha']], dim=-1)
        raw = torch.reshape(outputs_flat, list(pts.shape[:-1]) + [outputs_flat.shape[-1]])

        rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(raw, z_vals, rays_d, 0, True)

        if len(cfg.task_arg.cascade_samples)>1:

            rgb_map_0, disp_map_0, acc_map_0 = rgb_map, disp_map, acc_map

            z_vals_mid = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
            z_samples = sample_pdf(z_vals_mid,
                                   weights[..., 1:-1],
                                   128,
                                   det=(1 == 0.))
            z_samples = z_samples.detach()

            z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)
            pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[
                ..., :, None]  # [N_rays, N_samples + N_importance, 3]
            
            input_pts = pts.reshape(-1, C)
            input_viewdirs = viewdirs[:, None].expand(pts.shape).reshape(-1, 3)

            # raw = run_network(pts, fn=run_fn)
            ret = self.net(input_pts, input_viewdirs, model='fine')

            outputs_flat = torch.cat([ret['rgb'], ret['rawalpha']], dim=-1)
            raw = torch.reshape(outputs_flat, list(pts.shape[:-1]) + [outputs_flat.shape[-1]])

            rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(raw, z_vals, rays_d, 0, True)

        ret = {
            'rgb': rgb_map,
            'disp_map': disp_map,
            'acc': acc_map,
            'depth_map': depth_map
        }
        ret['raw'] = raw
        if len(cfg.task_arg.cascade_samples)>1:
            ret['rgb_coarse'] = rgb_map_0
            ret['disp0'] = disp_map_0
            ret['acc_coarse'] = acc_map_0
            ret['z_std'] = torch.std(z_samples, dim=-1, unbiased=False)  # [N_rays]

        for k in ret:
            DEBUG = False
            if (torch.isnan(ret[k]).any() or torch.isinf(ret[k]).any()) and DEBUG:
                print(f"! [Numerical Error] {k} contains nan or inf.")

        return ret


    def batchify_rays(self, rays_flat, chunk=1024 * 32):
        """Render rays in smaller minibatches to avoid OOM."""
        all_ret = {}
        for i in range(0, rays_flat.shape[0], chunk):
            ret = self.render_rays(rays_flat[i:i + chunk])
            for k in ret:
                if k not in all_ret:
                    all_ret[k] = []
                all_ret[k].append(ret[k])
        all_ret = {k: torch.cat(all_ret[k], 0) for k in all_ret}
        return all_ret


    def render(self, batch):
        rays_o = batch['rays_o']
        rays_d = batch['rays_d']
        near = batch['near']
        far = batch['far']

        sh = rays_o.shape
        # print(f"input shape: {sh}")
        rays_o, rays_d = rays_o.view(-1, 3), rays_d.view(-1, 3)
        near, far = near * torch.ones_like(rays_d[..., :-1]), far * torch.ones_like(rays_d[..., :-1])
        viewdirs = rays_d
        viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
        rays = torch.cat([rays_o, rays_d, near, far, viewdirs], dim=-1)
        ret = self.batchify_rays(rays, cfg.task_arg.chunk_size)
        ret = {k: v.view(*sh[:-1], -1) for k, v in ret.items()}
        ret['depth_map'] = ret['depth_map'].view(*sh[:-1])
        return ret

def raw2outputs(raw, z_vals, rays_d, raw_noise_std=0, white_bkgd=False):
    raw2alpha = lambda raw, dists, act_fn=F.relu: 1. - torch.exp(-act_fn(raw) * dists)

    dists = z_vals[..., 1:] - z_vals[..., :-1]
    dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists[..., :1].shape).to(dists)], -1)  # [N_rays, N_samples]

    dists = dists * torch.norm(rays_d[..., None, :], dim=-1)

    rgb = torch.sigmoid(raw[..., :3])  # [N_rays, N_samples, 3]
    noise = 0.
    if raw_noise_std > 0.:
        noise = torch.randn(raw[..., 3].shape) * raw_noise_std

    alpha = raw2alpha(raw[..., 3] + noise, dists)  # [N_rays, N_samples]
    weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)).to(alpha), 1. - alpha + 1e-10], -1), -1)[:, :-1]
    rgb_map = torch.sum(weights[..., None] * rgb, -2)  # [N_rays, 3]

    depth_map = torch.sum(weights * z_vals, -1)
    disp_map = 1. / torch.max(1e-10 * torch.ones_like(depth_map).to(depth_map), depth_map / torch.sum(weights, -1))
    acc_map = torch.sum(weights, -1)

    if white_bkgd:
        rgb_map = rgb_map + (1. - acc_map[..., None])

    return rgb_map, disp_map, acc_map, weights, depth_map


# Hierarchical sampling (section 5.2)
def sample_pdf(bins, weights, N_samples, det=False):
    # Get pdf
    weights = weights + 1e-5  # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)  # (batch, len(bins))

    # Take uniform samples
    if det:
        u = torch.linspace(0., 1., steps=N_samples).to(cdf)
        u = u.expand(list(cdf.shape[:-1]) + [N_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [N_samples]).to(cdf)

    # Invert CDF
    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, side='right')
    below = torch.max(torch.zeros_like(inds - 1), inds - 1)
    above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[..., 1] - cdf_g[..., 0])
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

    return samples
