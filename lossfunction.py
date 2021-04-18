from torch import nn
import torch.nn.functional as F
import torch

class BasicLoss(nn.Module):
    def __init__(self):
        super(BasicLoss, self).__init__()


    def forward(self, out_red, in_blue, im_restore, mask_red, mask_blue, mask_edge):
        red_restore  = F.max_pool2d(im_restore * mask_red, kernel_size=2, stride=2)
        blue_restore = F.max_pool2d(im_restore * mask_blue, kernel_size=2, stride=2)

        out_red = torch.clamp(out_red, 0, 1)
        red_restore = torch.clamp(red_restore, 0, 1)
        blue_restore = torch.clamp(blue_restore, 0, 1)

        mse =  (out_red - in_blue)**2 * mask_edge
        R =  (out_red - in_blue - red_restore + blue_restore)**2 * mask_edge

        return torch.mean(mse + R)