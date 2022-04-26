from torch import nn
import torch
from sklearn.metrics import accuracy_score

class L2(nn.Module):
    def __init__(self, weight, name="L2_img", metric= False):
        super().__init__()
        self.name = f"{name}_loss" if not metric else f"{name}_metric"
        self.weight = weight
        self.criterion = nn.MSELoss()
        
    def forward(self,inp, gt):
        return self.criterion(inp, gt)*self.weight
    
class L1(nn.Module):
    def __init__(self, weight, name="L1_img", metric= False):
        super().__init__()
        self.name = f"{name}_loss" if not metric else f"{name}_metric"
        self.weight = weight
        self.criterion = nn.L1Loss()
        
    def forward(self,inp, gt):
        return self.criterion(inp, gt) * self.weight
    
class CrossEntropy(nn.Module):
    def __init__(self, weight, name="CrossEntropy_cls", metric= False):
        super().__init__()
        self.name = f"{name}_loss" if not metric else f"{name}_metric"
        self.weight = weight
        self.criterion = nn.CrossEntropyLoss()
        
    def forward(self,inp, gt):
        return self.criterion(inp, gt) * self.weight

class Accuracy:
    def __init__(self, name="Accuracy_score"):
        super().__init__()
        self.criterion = accuracy_score
        self.name = name
    def __call__(self,inp, gt):
        _, preds = torch.max(inp, 1)
        return torch.sum(preds == gt) / gt.shape[0]
        

def compute_criterion(pred, gt, loss_list, metric_list):
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
            metrics[metric.name] = metric_val.cpu()
        
    return total_loss, losses, metrics
    
