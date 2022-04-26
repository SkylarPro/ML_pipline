#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import sys
import os
from functools import partial
import torch
from tqdm import tqdm

sys.path.append("/data/hdd1/brain/BraTS19/MILTestTasks")
from src.utils.CustomDataset import DatasetCIFAR100
from src.utils.criterions import (L1, L2, CrossEntropy,Accuracy,
                                  compute_criterion)
from src.models.model_loader import load_model
from src.utils.logs import AccumulativeDict,plot_params,res_to_string
from src.utils.mixin.config import Args


config_inf = {
    "log_dir": "logs",
}
cfg_model = {
   "name":"ResNetUnet",
   "count_cls":100,
   "number_channel": 3,
   "weights_path":"../src/train/logs/model/weight/ResNetUnet_mode_img_to_img_last.pt"
}
cfg_dataset = {
    "path_to_data": "../data",
    "bs": 16,
    "val_size": 0.2
}
class Inference:
    
    def __init__(self,config_inf = None, config_data = None, config_model = None):
        config_inf.update(config_model)
        self.args = Args(config_inf)
        self.cfg_model = Args(config_model)
        
        self.test_loader = DatasetCIFAR100(config_data).test_loader
        self.model = load_model(self.cfg_model)
        self.define_metric()
        self.compute_criterion_cls = partial(compute_criterion,
                                         loss_list = None,
                                         metric_list = self.metrics_cls
                                         )
                
    def define_metric(self,):
        self.metrics_cls = [Accuracy()]
        
    def apply(self,):
        apply_log = AccumulativeDict()
        self.model.cuda()
        with torch.no_grad():
            for i_step, data  in tqdm(enumerate(self.test_loader)):
                
                imgs_gpu = data[0].cuda()
                labels_gpu = data[1].cuda()
    
                prediction_img, prediction_cls = self.model(imgs_gpu)
                
                _, _, metric_log = self.compute_criterion_cls(
                                                    prediction_cls, labels_gpu,
                                                    )
                apply_log += metric_log
                                        
                del imgs_gpu
                del labels_gpu
        apply_log /= (i_step + 1)   
        print(res_to_string(apply_log,0,0))
        return apply_log
    
    
if __name__=="__main__":
    infernce = Inference(config_inf, cfg_dataset, cfg_model)
    infernce.apply()

