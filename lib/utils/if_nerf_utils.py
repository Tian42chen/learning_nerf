import numpy as np
import torch

# NOTE 真的是离大谱, nerf 的 transform matrix 是 c2w, 而不是通常说的 w2c
def get_rays_nerf(H, W, K, c2w):
    i, j = np.meshgrid(np.arange(W, dtype=np.float32),
                       np.arange(H, dtype=np.float32),
                       indexing='xy')
    dirs = np.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -np.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # rays_d = rays_d / np.linalg.norm(rays_d, axis=2, keepdims=True)
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = np.broadcast_to(c2w[:3,-1], rays_d.shape) # NOTE T 直接是 相机的世界坐标
    return rays_o, rays_d

def get_rays(H, W, K, R, T):
    # calculate the camera origin
    rays_o = -np.dot(R.T, T).ravel()
    # calculate the world coodinates of pixels
    i, j = np.meshgrid(np.arange(W, dtype=np.float32),
                       np.arange(H, dtype=np.float32),
                       indexing='xy')
    xy1 = np.stack([i, j, np.ones_like(i)], axis=2)
    pixel_camera = np.dot(xy1, np.linalg.inv(K).T)
    pixel_world = np.dot(pixel_camera - T.ravel(), R)
    # calculate the ray direction
    rays_d = pixel_world - rays_o[None, None]
    rays_d = rays_d / np.linalg.norm(rays_d, axis=2, keepdims=True)
    rays_o = np.broadcast_to(rays_o, rays_d.shape)
    return rays_o, rays_d

def extract_parameters(transform_matrix, focal, w, h):
    R = transform_matrix[:3, :3]
    T = transform_matrix[:3, 3]
    K = np.array([[focal, 0, w/2], [0, focal, h/2], [0, 0, 1]])
    return K, R, T

def render_weights(alpha: torch.Tensor, epsilon=1e-10):
    # alpha: N_rays, N_samples
    weights = alpha * torch.cumprod(torch.cat([alpha.new_ones(alpha.shape[0], 1), 1.-alpha + epsilon], dim=-1), dim=-1)[..., :-1]  # (N_rays, N_samples)
    return weights

def volume_rendering(rgb, alpha, epsilon=1e-8, bg_brightness=None, bg_image=None):
    # NOTE: here alpha's last dim is not 1, but N_samples
    # rgb: N_rays, N_samples, 3
    # alpha: N_rays, N_samples
    # bg_image: N_rays, 3 or None, if this is given as not None, the last sample on the ray will be replaced by this value (assuming this lies on the background)
    # We need to assume:
    # 1. network will find the True geometry, thus giving the background image its real value
    # 2. background image is rendered in a non-static fasion
    # returns:
    # weights: N_rays, N_samples
    # rgb_map: N_rays, 3
    # acc_map: N_rays,

    if bg_image is not None:
        rgb[:, :, -1] = bg_image

    weights = render_weights(alpha, epsilon)  # (N_rays, N_samples)
    rgb_map = torch.sum(weights[..., None] * rgb, dim=-2)  # (N_rays, 3)
    acc_map = torch.sum(weights, -1)  # (N_rays)

    if bg_brightness is not None:
        rgb_map = rgb_map + (1. - acc_map[..., None]) * bg_brightness

    return weights, rgb_map, acc_map

def get_5D_coords(rays_o, rays_d, z_vals, perturb):
    if perturb > 0.:# and self.net.training:
        # get intervals between samples
        mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
        upper = torch.cat([mids, z_vals[..., -1:]], -1)
        lower = torch.cat([z_vals[..., :1], mids], -1)
        # stratified samples in those intervals
        t_rand = torch.rand(z_vals.shape, device=upper.device, dtype=upper.dtype)
        z_vals = lower + (upper - lower) * t_rand

    pts = rays_o[:, None, :] + rays_d[:, None, :]  * z_vals [..., None] # (N_rays, N_samples, 3)
    viewdirs = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
    viewdirs = rays_d[:, None, :].expand(pts.shape) # (N_rays, N_samples, 3)

    # calculate dists for the opacity computation
    dists = z_vals[..., 1:] - z_vals[..., :-1]
    dists = torch.cat([dists, dists[..., -1:]], dim=-1)*torch.norm(rays_d[...,None,:], dim=-1) # (N_rays, N_samples)
    return pts, viewdirs, dists

# Hierarchical sampling (section 5.2)
def sample_pdf(bins, weights, N_importance, perturb=False, epsilon=1e-8):
    # Get pdf
    weights = weights + epsilon # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[...,:1]), cdf], -1)  # (N_rays, N_samples-1)

    # Take uniform samples
    if perturb>0: u = torch.rand(list(cdf.shape[:-1]) + [N_importance])
    else:
        u = torch.linspace(0., 1., steps=N_importance)
        u = u.expand(list(cdf.shape[:-1]) + [N_importance])
        
    # Invert CDF
    u = u.to(cdf.device).contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds-1), inds-1)
    above = torch.min((cdf.shape[-1]-1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (N_rays, N_importance, 2)

    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]] # (N_rays, N_importance, N_samples-1)
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[...,1]-cdf_g[...,0])
    denom = torch.where(denom<epsilon, torch.ones_like(denom), denom)
    t = (u-cdf_g[...,0])/denom
    samples = bins_g[...,0] + t * (bins_g[...,1]-bins_g[...,0]) # (N_rays, N_importance)

    return samples.detach()

def crop_center(H, W, fraction=0.5):
    start_H = int(H*(1-fraction))//2
    end_H = start_H + int(H*fraction)
    start_W = int(W*(1-fraction))//2
    end_W = start_W + int(W*fraction)
    return start_H, end_H, start_W, end_W
