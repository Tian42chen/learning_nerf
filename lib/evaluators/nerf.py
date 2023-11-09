import numpy as np
from lib.config import cfg
import os
import imageio
from lib.utils import img_utils
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import torch.nn.functional as F
import torch
import lpips
import imageio
from lib.utils import img_utils
import cv2
import json

class Evaluator:
    def __init__(self,):
        self.psnrs = []
        self.iter=0
        os.makedirs(cfg.result_dir, exist_ok=True)
        os.makedirs(cfg.result_dir + '/vis', exist_ok=True)

    def evaluate(self, output, batch):
        # assert image number = 1
        H, W = batch['meta']['H'].item(), batch['meta']['W'].item()
        pred_rgb = output['rgb'][0].reshape(H, W, 3).detach().cpu().numpy()
        gt_rgb = batch['rgb'][0].reshape(H, W, 3).detach().cpu().numpy()
        psnr_item = psnr(gt_rgb, pred_rgb, data_range=1.)
        self.psnrs.append(psnr_item)
        save_path = os.path.join(cfg.result_dir, f'vis/res{self.iter}.jpg')
        self.iter += 1
        hc_img=img_utils.horizon_concate(gt_rgb, pred_rgb)
        imageio.imwrite(save_path,(hc_img*255).astype(np.uint8))

    def summarize(self):
        ret = {}
        ret.update({'psnr': np.mean(self.psnrs)})
        print(ret)
        self.psnrs = []
        print('Save visualization results to {}'.format(cfg.result_dir))
        json.dump(ret, open(os.path.join(cfg.result_dir, 'metrics.json'), 'w'))
        return ret