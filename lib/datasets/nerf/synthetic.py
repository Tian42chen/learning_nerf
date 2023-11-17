import torch
import torch.utils.data as data
import numpy as np
import os
from lib.utils.if_nerf_utils import extract_parameters, get_rays, get_rays_nerf, crop_center
from lib.utils.debug_utils import save_img
from lib.config import cfg
from torchvision import transforms as T
import imageio
import json
import cv2
import tqdm

class Dataset(data.Dataset):
    def __init__(self, **kwargs):
        super(Dataset, self).__init__()
        data_root = kwargs['data_root']
        self.split = kwargs['split']
        input_ratio = kwargs['input_ratio']
        start, end, step = kwargs['cams']

        self.near = np.array(kwargs['near']).astype(np.float32)
        self.far = np.array(kwargs['far']).astype(np.float32)

        self.precrop_iters = 0
        if self.split == 'train' and 'precrop' in kwargs:
            self.precrop_iters = kwargs['precrop'].iters
            self.precrop_frac = kwargs['precrop'].frac

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
            self.camera_poses.append(np.array(camera_pose))

            image_path = os.path.join(self.data_root, frame
            ['file_path'][2:] + '.png')
            image = imageio.imread(image_path)
            if input_ratio != 1.:
                image = cv2.resize(image, None, fx=input_ratio, fy=input_ratio, interpolation=cv2.INTER_AREA)
            self.images.append(np.array(image))

        self.images = (np.array(self.images)/255.).astype(np.float32)
        self.camera_poses = np.array(self.camera_poses).astype(np.float32)

        if cfg.task_arg.white_bkgd:
            self.images = self.images[..., :3] * self.images[..., -1:] + (1 - self.images[..., -1:])
        else:
            self.images = self.images[..., :3]

        self.H, self.W = self.images[0].shape[:2]

        camera_angle_x = float(json_info['camera_angle_x'])
        focal = .5 * self.W / np.tan(.5 * camera_angle_x)
        self.K = np.array([[focal, 0, self.W/2], [0, focal, self.H/2], [0, 0, 1]])

        if 'perturb' in kwargs:
            self.perturb = kwargs['perturb']
        else:
            self.perturb = 0
    
    def __getitem__(self, index):
        image = self.images[index]
        camera_pose = self.camera_poses[index]

        rays_o, rays_d = get_rays_nerf(self.H, self.W, self.K, camera_pose)
        rays_o, rays_d = rays_o.astype(np.float32), rays_d.astype(np.float32)

        if cfg.debug:
            save_img(image, f'img{index}', time=True)
            # print("rays_o: ", rays_o.shape)
            # print(f"R.T{index}: ", R.T)
            # print(f"T{index}: ", T)
            # print(f"R.T*T{index}: ", np.dot(R.T, T))
            # print(f"camera_pose{index}: ", -np.dot(R.T, T).ravel())
            # print(f"data{index}: ", rays_o[0][0])
            # print("\n")

        ret={}
        if self.split == 'train':
            if self.precrop_iters > 0:
                self.precrop_iters -= 1
                start_H, end_H, start_W, end_W = crop_center(self.H, self.W, self.precrop_frac)
                coords = np.stack(
                    np.meshgrid(
                        np.arange(start_H, end_H),
                        np.arange(start_W, end_W)
                    ), -1)
            else: 
                coords = np.stack(np.meshgrid(np.arange(self.H), np.arange(self.W)), -1)
                
            coords = coords.reshape(-1, 2)
            ids = np.random.choice(coords.shape[0], self.batch_size, replace=False)
            ids = coords[ids]

            if cfg.debug:
                save_img(image, f'crop_img{index}', time=True)
                # print("HW: ", HW)
                # print("rays_o: ", rays_o.shape)

            ret.update({
                'rays_o': rays_o[ids[:,0], ids[:,1]], 
                'rays_d': rays_d[ids[:,0], ids[:,1]],
                'rgb': image[ids[:,0], ids[:,1]]
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