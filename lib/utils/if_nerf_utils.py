import numpy as np
import torch


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
    # alpha: n_rays, n_samples
    weights = alpha * torch.cumprod(torch.cat([alpha.new_ones(alpha.shape[0], 1), 1.-alpha + epsilon], dim=-1), dim=-1)[..., :-1]  # (n_rays, n_samples)
    return weights


def volume_rendering(rgb, alpha, epsilon=1e-8, bg_brightness=None, bg_image=None):
    # NOTE: here alpha's last dim is not 1, but n_samples
    # rgb: n_rays, n_samples, 3
    # alpha: n_rays, n_samples
    # bg_image: n_rays, 3 or None, if this is given as not None, the last sample on the ray will be replaced by this value (assuming this lies on the background)
    # We need to assume:
    # 1. network will find the True geometry, thus giving the background image its real value
    # 2. background image is rendered in a non-static fasion
    # returns:
    # weights: n_rays, n_samples
    # rgb_map: n_rays, 3
    # acc_map: n_rays,

    if bg_image is not None:
        rgb[:, :, -1] = bg_image

    weights = render_weights(alpha, epsilon)  # (n_rays, n_samples)
    rgb_map = torch.sum(weights[..., None] * rgb, dim=-2)  # (n_rays, 3)
    acc_map = torch.sum(weights, -1)  # (n_rays)

    if bg_brightness is not None:
        rgb_map = rgb_map + (1. - acc_map[..., None]) * bg_brightness

    return weights, rgb_map, acc_map