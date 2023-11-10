import time
# import open3d as o3d
import os
import numpy as np
import torch
from lib.config import cfg

def get_pre(name):
    if not os.path.exists('debug'): 
        os.makedirs('debug')
    if not os.path.exists(f'debug/{cfg.exp_name}'): 
        os.makedirs(f'debug/{cfg.exp_name}')
    return f"debug/{cfg.exp_name}/{name}"

def get_time():
    current_time = time.localtime()
    month = str(current_time.tm_mon).zfill(2)
    day = str(current_time.tm_mday).zfill(2)
    hour = str(current_time.tm_hour).zfill(2)
    minute = str(current_time.tm_min).zfill(2)
    second= str(current_time.tm_sec).zfill(2)
    return month + day + '-' + hour + ':' + minute + ':' + second

def to_numpy(a)->np.ndarray:
    if type(a) == torch.Tensor:
        if a.is_cuda:
            a = a.cpu()
        return a.detach().numpy()
    elif type(a) == np.ndarray:
        return a
    else:
        raise TypeError('Unsupported data type')

def save_point_cloud(point_cloud, filename):
    if not cfg.debug: return
    return
    point_cloud = to_numpy(point_cloud)
    
    # 将numpy数组转换为open3d的PointCloud对象
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)

    # 保存PointCloud到文件
    o3d.io.write_point_cloud(get_pre(filename), pcd)

def output_debug_log(output, name):
    if not cfg.debug: return
    if type(output) != str:
        output = str(output)

    with open(f"{get_pre(name)}.log", 'w') as f:
        f.write(output)

def save_debug(a, name):
    if not cfg.debug: return
    np.save( f'{get_pre(name)}.npy', to_numpy(a))