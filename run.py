from zmq import Flag
from lib.config import cfg, args
import numpy as np
import os
### SCRIPTS BEGINING ###
def run_dataset():
    from lib.datasets import make_data_loader
    import tqdm

    cfg.train.num_workers = 0
    data_loader = make_data_loader(cfg, is_train=True)
    for batch in tqdm.tqdm(data_loader):
        pass

def run_network():
    from lib.networks import make_network
    from lib.datasets import make_data_loader
    from lib.networks.renderers import make_renderer
    from lib.utils.net_utils import load_network
    from lib.utils.data_utils import to_cuda
    import tqdm
    import torch
    import time

    network = make_network(cfg).cuda()
    load_network(network, cfg.trained_model_dir, epoch=cfg.test.epoch, resume=cfg.resume)
    network.eval()

    i=0
    data_loader = make_data_loader(cfg, is_train=True)
    renderer = make_renderer(cfg, network)
    total_time = 0
    for batch in tqdm.tqdm(data_loader):
        batch = to_cuda(batch)
        with torch.no_grad():
            torch.cuda.synchronize()
            start = time.time()
            renderer.render(batch)
            torch.cuda.synchronize()
            total_time += time.time() - start
        if cfg.debug and i >= 0 : break
        i +=1
    print(total_time / len(data_loader))


def run_evaluate():
    from lib.datasets import make_data_loader
    from lib.networks.renderers import make_renderer
    from lib.evaluators import make_evaluator
    import tqdm
    import torch
    from lib.networks import make_network
    from lib.utils.net_utils import load_network
    import time

    cfg.resume=True
    
    network = make_network(cfg).cuda()
    load_network(network, cfg.trained_model_dir, resume=cfg.resume, epoch=cfg.test.epoch)
    network.eval()

    data_loader = make_data_loader(cfg, is_train=False)
    renderer = make_renderer(cfg, network)
    evaluator = make_evaluator(cfg)
    net_time = []
    for batch in tqdm.tqdm(data_loader):
        for k in batch:
            if k != 'meta':
                batch[k] = batch[k].cuda()
        with torch.no_grad():
            torch.cuda.synchronize()
            start_time = time.time()
            output = renderer.render(batch)
            torch.cuda.synchronize()
            end_time = time.time()
        net_time.append(end_time - start_time)
        evaluator.evaluate(output, batch)
    evaluator.summarize()
    if len(net_time) > 1:
        print('net_time: ', np.mean(net_time[1:]))
        print('fps: ', 1./np.mean(net_time[1:]))
    else:
        print('net_time: ', np.mean(net_time))
        print('fps: ', 1./np.mean(net_time))

def run_visualize():
    # return 
    from lib.networks import make_network
    from lib.datasets import make_data_loader
    from lib.utils.net_utils import load_network
    from lib.utils import net_utils
    import tqdm
    import torch
    from lib.visualizers import make_visualizer
    from lib.utils.data_utils import to_cuda

    cfg.resume=True

    network = make_network(cfg).cuda()
    load_network(network,
                 cfg.trained_model_dir,
                 resume=cfg.resume,
                 epoch=cfg.test.epoch)
    network.eval()

    # data_loader = make_data_loader(cfg, is_train=False)
    visualizer = make_visualizer(cfg)
    # for batch in tqdm.tqdm(data_loader):
    #     batch = to_cuda(batch)
    #     with torch.no_grad():
    #         output = network(batch)
    #     visualizer.visualize(output, batch)
    if visualizer.write_video:
        visualizer.summarize()

if __name__ == '__main__':
    globals()['run_' + args.type]()
