from torch import nn
import torch
from sklearn.metrics import accuracy_score

import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from math import exp

from typing import Tuple, Dict,List

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def _ssim(img1, img2, window, window_size, channel, size_average = True):
    mu1 = F.conv2d(img1, window, padding = window_size//2, groups = channel)
    mu2 = F.conv2d(img2, window, padding = window_size//2, groups = channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))
    if size_average:
        return ssim_map
    else:
        return ssim_map.mean(1).mean(1).mean(1)

class SSIM(torch.nn.Module):
    def __init__(self,weight=1.0, name="SSIM", window_size = 11, size_average = True, metric=False):
        super(SSIM, self).__init__()
        self.name = f"{name}_loss" if not metric else f"{name}_metric"
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 3
        self.metric = metric
        self.weight = weight
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)
            
            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)
            
            self.window = window
            self.channel = channel
        
        
        ssim_map = _ssim(img1, img2, window, self.window_size, channel, self.size_average)
        if not self.metric:
            return torch.clip(0.5*(1.0 - ssim_map),0.0, 1.0).mean()
        return torch.clip(0.5*(1.0 + ssim_map),0.0, 1.0).mean()

class L2(nn.Module):
    def __init__(self, weight=1.0, name="L2_img", metric= False):
        super().__init__()
        self.name = f"{name}_loss" if not metric else f"{name}_metric"
        self.weight = weight
        self.criterion = nn.MSELoss()
        
    def forward(self,inp, gt):
        return self.criterion(inp, gt)*self.weight
    
class L1(nn.Module):
    def __init__(self, weight=1.0, name="L1_img", metric= False):
        super().__init__()
        self.name = f"{name}_loss" if not metric else f"{name}_metric"
        self.weight = weight
        self.criterion = nn.L1Loss()
        
    def forward(self,inp, gt):
        return self.criterion(inp, gt) * self.weight
    
class CrossEntropy(nn.Module):
    def __init__(self, weight=1.0, name="CrossEntropy_cls", metric= False):
        super().__init__()
        self.name = f"{name}_loss" if not metric else f"{name}_metric"
        self.weight = weight
        self.criterion = nn.CrossEntropyLoss()
        
    def forward(self,inp, gt):
        return self.criterion(inp, gt) * self.weight

class Accuracy:
    def __init__(self, name="Accuracy_score") -> int:
        super().__init__()
        self.criterion = accuracy_score
        self.name = name
    def __call__(self,inp, gt):
        _, preds = torch.max(inp, 1)
        return torch.sum(preds == gt) / gt.shape[0]
    
class Accum_prediction:
    def __init__(self, name="Accum_pred"):
        super().__init__()
        self.name = name
    def __call__(self,inp, gt) -> List[int]:
        _, preds = torch.max(inp, 1)
        return preds.tolist()
    
class Accum_gt:
    def __init__(self, name="Accum_gt"):
        super().__init__()
        self.name = name
    def __call__(self,inp, gt) -> List[int]:
        return gt.tolist()

def compute_criterion(pred, gt, loss_list, metric_list) -> Tuple[int,Dict,Dict]:
    losses = {}
    metrics = {}
    total_loss = 0
    if loss_list:
        for loss in loss_list:
            loss_val = loss(pred, gt)
            losses[loss.name] = loss_val.cpu()
            total_loss += loss_val*loss.weight
    if metric_list:
        for metric in metric_list:
            metric_val = metric(pred, gt)
            metrics[metric.name] = metric_val.detach().cpu() if not isinstance(metric_val, list) else metric_val
        
    return total_loss, losses, metrics
    
