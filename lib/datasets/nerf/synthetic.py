import torch
import torch.utils.data as data
import numpy as np
import os
from lib.utils.if_nerf_utils import extract_parameters, get_rays
from lib.config import cfg
from torchvision import transforms as T
import imageio
import json
import cv2

class Dataset(data.Dataset):
    def __init__(self, **kwargs):
        super(Dataset, self).__init__()
        data_root = kwargs['data_root']
        self.split = kwargs['split']
        input_ratio = kwargs['input_ratio']
        start, end, step = kwargs['cams']

        self.near = kwargs['near']
        self.far = kwargs['far']

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
        self.focal = .5 * self.W / np.tan(.5 * camera_angle_x)
    
    def __getitem__(self, index):
        image = self.images[index]
        camera_pose = self.camera_poses[index]

        K, R, T = extract_parameters(camera_pose, self.focal, self.W, self.H)

        rays_o, rays_d = get_rays(self.H, self.W, K, R, T)
        rays_o, rays_d = rays_o.astype(np.float32), rays_d.astype(np.float32)

        ret={}
        if self.split == 'train':
            ids = np.random.randint(0, self.H * self.W, size=self.batch_size)

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
            'near': np.broadcast_to(self.near, ret['rays_o'].shape[:-1] + (1,)).astype(np.float32),
            'far': np.broadcast_to(self.far, ret['rays_o'].shape[:-1] + (1,)).astype(np.float32)
        })

        ret.update({'meta': {'H': self.H, 'W': self.W}})
        return ret

    def __len__(self):
        return len(self.images)