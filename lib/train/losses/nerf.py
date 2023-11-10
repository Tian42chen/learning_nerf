import torch
import torch.nn as nn
from lib.utils import net_utils
from lib.config import cfg

class NetworkWrapper(nn.Module):
    def __init__(self, net):
        super(NetworkWrapper, self).__init__()
        self.net=net
        self.color_crit=nn.MSELoss(reduction='mean')
        # self.img2mse = lambda x, y : torch.mean((x - y) ** 2)
        self.mse2psnr=lambda x:-10.*torch.log(x.detach())/torch.log(torch.Tensor([10.]).to(x.device))

    def forward(self,batch):
        output=self.net(batch)

        scalar_stats={}
        loss=0
        color_loss=self.color_crit(output['rgb'],batch['rgb'])
        # color_loss=self.img2mse(output['rgb'],batch['rgb'])
        scalar_stats.update({'color_mse':color_loss})
        loss+=color_loss

        psnr=self.mse2psnr(color_loss)
        # scalar_stats.update({'psnr':psnr})

        scalar_stats.update({'acc': output['acc'].mean()})

        if 'rgb_coarse' in output:
            color_loss_=self.color_crit(output['rgb_coarse'],batch['rgb'])
            # color_loss_=self.img2mse(output['rgb_coarse'],batch['rgb'])
            scalar_stats.update({'color_mse_coarse':color_loss_})
            loss+=color_loss_
            psnr_=self.mse2psnr(color_loss_)
            # scalar_stats.update({'psnr_coarse':psnr_})

            scalar_stats.update({'acc_coarse': output['acc_coarse'].mean()})

        scalar_stats.update({'loss':loss})
        image_stats={}
        return output,loss,scalar_stats,image_stats