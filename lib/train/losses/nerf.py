import torch
import torch.nn as nn
from lib.utils import net_utils
from lib.networks.renderers import make_renderer
from lib.config import cfg

class NetworkWrapper(nn.Module):
    def __init__(self, net):
        super(NetworkWrapper, self).__init__()
        self.net = net
        self.renderer = make_renderer(cfg, self.net)
        self.color_crit=nn.MSELoss(reduction='mean')
        self.mse2psnr=lambda x:-10.*torch.log(x.detach())/torch.log(torch.Tensor([10.]).to(x.device))

    def forward(self,batch):
        output=self.renderer.render(batch)

        scalar_stats={}
        loss=0
        color_loss=self.color_crit(output['rgb'],batch['rgb'])
        scalar_stats.update({'color_mse':color_loss})
        loss+=color_loss

        psnr=self.mse2psnr(color_loss)
        scalar_stats.update({'psnr':psnr})

        scalar_stats.update({'acc': output['acc'].mean()})

        if 'rgb_coarse' in output:
            color_loss_coarse=self.color_crit(output['rgb_coarse'],batch['rgb'])
            scalar_stats.update({'color_mse_coarse':color_loss_coarse})
            loss+=color_loss_coarse
            psnr_coarse=self.mse2psnr(color_loss_coarse)
            scalar_stats.update({'psnr_coarse':psnr_coarse})

            scalar_stats.update({'acc_coarse': output['acc_coarse'].mean()})

        if cfg.debug:
            print("batch rgb: ", batch['rgb'].shape, batch['rgb'].mean().item())
            for key in output:
                print(f"{key}: ", output[key].shape, output[key].mean().item())

            # from torchviz import make_dot
            # dot = make_dot(loss)
            # dot.render(filename='exp/nerf', view=False, format='pdf') 

        scalar_stats.update({'loss':loss})
        image_stats={}
        return output,loss,scalar_stats,image_stats