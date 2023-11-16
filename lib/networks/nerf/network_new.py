import torch
import torch.nn as nn
from torch.nn import functional as F
from lib.networks.encoding import get_encoder
from lib.config import cfg
import numpy as np


class Network(nn.Module):
    def __init__(self,):
        super(Network, self).__init__()
        self.net = Network_inside()


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

        if net_c is None:
            raw = self.net(pts, viewdirs)
        else:
            raw = self.net(pts, viewdirs, net_c)
        rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(raw, z_vals, rays_d, 0, True)

        if 128 > 0:

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

            # raw = run_network(pts, fn=run_fn)
            if net_c is None:
                raw = self.net(pts, viewdirs, model='fine')
            else:
                raw = self.net(pts, viewdirs, net_c, model='fine')

            rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(raw, z_vals, rays_d, 0, True)

        ret = {
            'rgb': rgb_map,
            'disp_map': disp_map,
            'acc': acc_map,
            'depth_map': depth_map
        }
        ret['raw'] = raw
        if 128 > 0:
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


    def forward(self, batch):
        rays_o = batch['rays_o']
        rays_d = batch['rays_d']
        near = batch['near']
        far = batch['far']

        sh = rays_o.shape
        # print(f"input shape: {sh}")
        rays_o, rays_d = rays_o.view(-1, 3), rays_d.view(-1, 3)
        near , far = near.view(-1, 1), far.view(-1, 1)
        # near, far = near * torch.ones_like(rays_d[..., :-1]), far * torch.ones_like(rays_d[..., :-1])
        viewdirs = rays_d
        viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
        rays = torch.cat([rays_o, rays_d, near, far, viewdirs], dim=-1)
        ret = self.batchify_rays(rays, cfg.task_arg.chunk_size)
        ret = {k: v.view(*sh[:-1], -1) for k, v in ret.items()}
        ret['depth_map'] = ret['depth_map'].view(*sh[:-1])
        return ret
    


class NeRF(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, input_ch_views=3, skips=[], use_viewdirs=False):
        super(NeRF, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.skips = skips
        self.use_viewdirs = use_viewdirs
        self.output_ch = 5 if self.use_viewdirs else 4

        self.pts_linears = nn.ModuleList(
        [nn.Linear(self.input_ch, self.W)] + [nn.Linear(self.W, self.W) if i not in self.skips else nn.Linear(self.W + self.input_ch, self.W) for i in
                                        range(self.D - 1)])

        self.views_linears = nn.ModuleList([nn.Linear(self.input_ch_views + self.W, self.W // 2)])

        if self.use_viewdirs:
            # feature vector(256)
            self.feature_linear = nn.Linear(self.W, self.W)
            # alpha(1)
            self.alpha_linear = nn.Linear(self.W, 1)
            # rgb color(3)
            self.rgb_linear = nn.Linear(self.W // 2, 3)
        else:
            # output channel(default: 4)
            self.output_linear = nn.Linear(self.W, self.output_ch)


    def forward(self, x):
        input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)
        h = input_pts
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

        if self.use_viewdirs:
            alpha = self.alpha_linear(h)
            feature = self.feature_linear(h)
            h = torch.cat([feature, input_views], -1)

            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                h = F.relu(h)

            rgb = self.rgb_linear(h)
            outputs = torch.cat([rgb, alpha], -1)
        else:
            outputs = self.output_linear(h)

        return outputs


    def load_weights_from_keras(self, weights):
        assert self.use_viewdirs, "Not implemented if use_viewdirs=False"

        # Load pts_linears
        for i in range(self.D):
            idx_pts_linears = 2 * i
            self.pts_linears[i].weight.data = torch.from_numpy(
                np.transpose(weights[idx_pts_linears]))
            self.pts_linears[i].bias.data = torch.from_numpy(
                np.transpose(weights[idx_pts_linears + 1]))

        # Load feature_linear
        idx_feature_linear = 2 * self.D
        self.feature_linear.weight.data = torch.from_numpy(
            np.transpose(weights[idx_feature_linear]))
        self.feature_linear.bias.data = torch.from_numpy(
            np.transpose(weights[idx_feature_linear + 1]))

        # Load views_linears
        idx_views_linears = 2 * self.D + 2
        self.views_linears[0].weight.data = torch.from_numpy(
            np.transpose(weights[idx_views_linears]))
        self.views_linears[0].bias.data = torch.from_numpy(
            np.transpose(weights[idx_views_linears + 1]))

        # Load rgb_linear
        idx_rbg_linear = 2 * self.D + 4
        self.rgb_linear.weight.data = torch.from_numpy(
            np.transpose(weights[idx_rbg_linear]))
        self.rgb_linear.bias.data = torch.from_numpy(
            np.transpose(weights[idx_rbg_linear + 1]))

        # Load alpha_linear
        idx_alpha_linear = 2 * self.D + 6
        self.alpha_linear.weight.data = torch.from_numpy(
            np.transpose(weights[idx_alpha_linear]))
        self.alpha_linear.bias.data = torch.from_numpy(
            np.transpose(weights[idx_alpha_linear + 1]))


class Network_inside(nn.Module):
    def __init__(self):
        super(Network_inside, self).__init__()
        self.N_samples = 64
        self.N_importance = 128
        self.chunk = cfg.task_arg.chunk_size
        self.batch_size = cfg.task_arg.N_rays
        self.white_bkgd = cfg.task_arg.white_bkgd
        self.use_viewdirs = True
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # encoder
        self.embed_fn, self.input_ch = get_encoder(cfg.network.xyz_encoder)
        self.embeddirs_fn, self.input_ch_views = get_encoder(cfg.network.dir_encoder)

        # coarse model
        self.model = NeRF(D=cfg.network.nerf.D,
                          W=cfg.network.nerf.W,
                          input_ch=self.input_ch,
                          input_ch_views=self.input_ch_views,
                        #   skips=cfg.network.nerf.skips,
                          use_viewdirs=self.use_viewdirs)

        # fine model
        self.model_fine = NeRF(D=cfg.network.nerf.D,
                               W=cfg.network.nerf.W,
                               input_ch=self.input_ch,
                               input_ch_views=self.input_ch_views,
                            #    skips=cfg.network.nerf.skips,
                               use_viewdirs=self.use_viewdirs)


    def batchify(self, fn, chunk):
        """Constructs a version of 'fn' that applies to smaller batches."""
        def ret(inputs):
            return torch.cat([fn(inputs[i:i + chunk]) for i in range(0, inputs.shape[0], chunk)], 0)
        return ret


    def forward(self, inputs, viewdirs, model=''):
        """Prepares inputs and applies network 'fn'."""
        if model == 'fine':
            fn = self.model_fine
        else:
            fn = self.model

        inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])
        embedded = self.embed_fn(inputs_flat)

        if self.use_viewdirs:
            input_dirs = viewdirs[:, None].expand(inputs.shape)
            input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]])
            embedded_dirs = self.embeddirs_fn(input_dirs_flat)
            embedded = torch.cat([embedded, embedded_dirs], -1)

        embedded = embedded.to(torch.float32)
        outputs_flat = self.batchify(fn, self.chunk)(embedded)
        outputs = torch.reshape(outputs_flat, list(inputs.shape[:-1]) + [outputs_flat.shape[-1]])
        return outputs


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

    # if raw[...,3].max() <0.:
    #     print("rawalpha: ", raw[...,3].mean())
    #     print("rawalpha: ", raw[...,3].max())
    #     print("alpha: ", alpha.mean())
    #     print("alpha: ", alpha.max())
    #     raise ValueError("alpha.mean() <0.")

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
