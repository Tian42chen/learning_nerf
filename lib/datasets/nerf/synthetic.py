import torch
import torch.utils.data as data
import numpy as np
import os
from lib.utils.if_nerf_utils import extract_parameters, get_rays, get_rays_nerf, crop_center, Precrop_Counter
from lib.utils.debug_utils import save_img
from lib.config import cfg
from torchvision import transforms as T
import imageio
import json
import cv2
import tqdm

class Dataset(data.Dataset):
    def __init__(self, data_cfg):
        super(Dataset, self).__init__()
        data_root = data_cfg.data_root
        self.split = data_cfg.split
        input_ratio = data_cfg.input_ratio
        start, end, step = data_cfg.cams

        self.precrop_counter = None
        if 'precrop' in data_cfg:
            self. precrop_counter = Precrop_Counter(data_cfg.precrop, cfg.record_dir)

        self.near = np.array(data_cfg.near).astype(np.float32)
        self.far = np.array(data_cfg.far).astype(np.float32)

        scene = cfg.scene
        self.data_root = os.path.join(data_root, scene)
        self.batch_size = cfg.task_arg.N_rays

        json_info = json.load(open(os.path.join(self.data_root, f'transforms_{self.split}.json')))

        info_length = len(json_info['frames'])
        if end <= -1 or info_length <= end: end = info_length

        # read data
        self.images = []
        self.camera_poses = []
        for frame in json_info['frames'][start:end:step]:
            camera_pose = frame['transform_matrix']
            self.camera_poses.append(np.array(camera_pose).astype(np.float32))

            image_path = os.path.join(self.data_root, frame
            ['file_path'][2:] + '.png')
            image = imageio.imread(image_path)/255.
            if cfg.task_arg.white_bkgd:
                image = image[..., :3] * image[..., -1:] + (1 - image[..., -1:])
            else:
                image = image[..., :3]

            if input_ratio != 1.:
                image = cv2.resize(image, None, fx=input_ratio, fy=input_ratio, interpolation=cv2.INTER_AREA)
            self.images.append(np.array(image).astype(np.float32))
            
        self.H, self.W = self.images[0].shape[:2]

        camera_angle_x = float(json_info['camera_angle_x'])
        focal = .5 * self.W / np.tan(.5 * camera_angle_x)
        self.K = np.array([[focal, 0, self.W/2], [0, focal, self.H/2], [0, 0, 1]])

        if 'perturb' in data_cfg:
            self.perturb = data_cfg.perturb
        else:
            self.perturb = 0
    
    def __getitem__(self, index):
        image = self.images[index]
        camera_pose = self.camera_poses[index]

        rays_o, rays_d = get_rays_nerf(self.H, self.W, self.K, camera_pose)
        rays_o, rays_d = rays_o.astype(np.float32), rays_d.astype(np.float32)

        ret={}
        if self.split == 'train':
            HW = self.H * self.W
            if self.precrop_counter is not None and self.precrop_counter():
                start_H, end_H, start_W, end_W = crop_center(self.H, self.W, self.precrop_counter.frac)
                HW = (end_H - start_H) * (end_W - start_W)
                image = image[start_H:end_H, start_W:end_W]
                rays_o = rays_o[start_H:end_H, start_W:end_W]
                rays_d = rays_d[start_H:end_H, start_W:end_W]

            ids = np.random.choice(HW, size=self.batch_size, replace=False)

            ret.update({
                'rays_o': rays_o.reshape(-1,3)[ids],
                'rays_d': rays_d.reshape(-1,3)[ids],
                'rgb': image.reshape(-1,3)[ids]
            })
        elif self.split == 'test':
            ret.update({
                'rays_o': rays_o.reshape(-1,3),
                'rays_d': rays_d.reshape(-1,3),
                'rgb': image.reshape(-1,3)
            })

        ret.update({
            'near': self.near,
            'far': self.far
        })

        meta = {'H': self.H, 'W': self.W}
        if self.perturb>0: meta.update({'perturb': self.perturb})

        ret.update({'meta': meta})
        return ret

    def __len__(self):
        return len(self.images)