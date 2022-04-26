#!/usr/bin/env python
# coding: utf-8

# In[2]:


import sys
import os
from functools import partial


sys.path.append("/data/hdd1/brain/BraTS19/MILTestTasks")
from src.utils.mixin.config import Args
from src.utils.CustomDataset import DatasetCIFAR100
from src.utils.logs import AccumulativeDict,plot_params,res_to_string
from src.utils.criterions import (L1, L2, CrossEntropy,Accuracy,
                                  compute_criterion)
from src.models.model_loader import load_model

import torch.nn as nn
import numpy as np
import torch
from torch.optim import Adam, lr_scheduler
from torch.utils.tensorboard import SummaryWriter


from typing import List,Tuple
from tqdm import tqdm


# mode [img_img or img_to_cls, combinate]
config_train = {
    "mode":"combinate",
    "lr": 1e-03,
    "weight_decay": 1e-05,
    "log_dir": "logs",
    "T_0" : 10,
    "T_mult": 2,
    "eta_min": 1e-09,
    "n_epoch": 5, 
    "bs":8,
    "p_save_model":"logs/model/weight"
}
cfg_dataset = {
    "path_to_data": "../../data",
    "bs": 16,
    "val_size": 0.2
}
cfg_model = {
   "name":"ResNetUnet",
   "count_cls":100,
   "number_channel": 3,
   "weights_path":"../train/logs/model/weight/ResNetUnet_mode_img_to_img_last.pt"
}

class TrainModel: 
    def __init__(self,config_train = None, config_data = None, config_model = None):
        config_train.update(config_model)
        self.args = Args(config_train)
        self.cfg_model = Args(config_model)
        
        self.dataset = DatasetCIFAR100(config_data)
        self.model = load_model(self.cfg_model) #load_model(path_to_model=self.args.weights) # add mode
        if not os.path.isdir(self.args.p_save_model):
            os.makedirs(self.args.p_save_model)
        self.define_loss()
        self.define_metric()
        
        self.optimizer = Adam([param for param in self.model.parameters() if param.requires_grad],
                              lr = self.args.lr, weight_decay = self.args.weight_decay
                             )
        self.scheduler = lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer,
                                                      T_0=self.args.T_0, 
                                                      T_mult=self.args.T_mult,
                                                      eta_min=self.args.eta_min)
        
        self.compute_criterion_img = partial(compute_criterion,
                                         loss_list = self.losses_img,
                                         metric_list = self.metrics_img
                                         )
        self.compute_criterion_cls = partial(compute_criterion,
                                         loss_list = self.losses_cls,
                                         metric_list = self.metrics_cls
                                         )
        
        if self.args.log_dir:
            self.writer_train = SummaryWriter(self.args.log_dir + "/train")
            self.writer_val = SummaryWriter(self.args.log_dir + "/val")
            
            
    def define_loss(self,):
        if self.args.mode == "img_to_img":
            self.losses_img = [
                L1(weight=1.0),
             ]
            self.losses_cls = None
        elif self.args.mode == "img_to_cls":
            self.losses_img = None
            self.losses_cls = [
                CrossEntropy(weight=1.0)
            ]
        elif self.args.mode == "combinate":
            self.losses_img = [
                L1(weight=1.0)
            ]
            self.losses_cls = [
                 CrossEntropy(weight=1.0)
            ]
        else:
            assert False, f"Mode should img_to_img, img_to_cls or combinate not {self.args.mode}"
    def define_metric(self,):
        self.metrics_img = [L2(weight=1.0, metric = True)]
        self.metrics_cls = [Accuracy()]
        
    def train_model(self,):    
        self.model.cuda()
        for epoch in range(self.args.n_epoch):
            
            train_log = AccumulativeDict()            
            self.model.train()
            for i_step, data  in tqdm(enumerate(self.dataset.train_loader)):
                
                    imgs_gpu = data[0].cuda()
                    labels_gpu = data[1].cuda()
                    
                    prediction_img, prediction_cls = self.model(imgs_gpu)
                    
                    total_loss_img, loss_log, metric_log = self.compute_criterion_img(
                                                        prediction_img.cpu(), imgs_gpu.cpu(),
                                                        )
                    train_log+=loss_log
                    train_log+=metric_log
                    total_loss_cls, loss_log, metric_log = self.compute_criterion_cls(
                                                        prediction_cls.cpu(), labels_gpu.cpu(),
                                                        )
                    train_log += loss_log
                    train_log += metric_log
                    
                    total_loss = total_loss_img + total_loss_cls
                    self.optimizer.zero_grad()
                    total_loss.backward()
                    self.optimizer.step()                     
                    del imgs_gpu
                    del metric_log
                    
            train_log /= (i_step + 1)
            train_log["LR"] = self.scheduler.get_last_lr()[-1]
            torch.save(self.model.state_dict(),f"{self.args.p_save_model}/{self.model.name}_mode_{self.args.mode}_last.pt")
            
            print("Train: ",res_to_string(train_log, epoch, self.args.n_epoch))
            
            val_log = self.compute_valid()
            
            print("Val: ",res_to_string(val_log, epoch, self.args.n_epoch))

            self.scheduler.step(epoch)
            
            plot_params(train_log, self.writer_train, epoch)
            plot_params(val_log, self.writer_val, epoch)
            
        self.writer_val.close()
        self.writer_train.close()
        return self.model

    def compute_valid(self):
        self.model.cuda()
        self.model.eval()
        
        val_log = AccumulativeDict()
        with torch.no_grad():
            for i_step, data  in tqdm(enumerate(self.dataset.val_loader)):
                
                imgs_gpu = data[0].cuda()
                labels_gpu = data[1].cuda()
    
                prediction_img, prediction_cls = self.model(imgs_gpu)
                
                _, loss_log, metric_log = self.compute_criterion_img(
                                                        prediction_img, imgs_gpu,
                                                        )
                val_log += loss_log
                val_log += metric_log
                _, loss_log, metric_log = self.compute_criterion_cls(
                                                    prediction_cls, labels_gpu,
                                                    )
                val_log += loss_log
                val_log += metric_log
                                        
                del imgs_gpu
                del labels_gpu
                
        val_log /= (i_step + 1)        
        return val_log
    
if __name__=="__main__":
    train_pipline = TrainModel(config_train, cfg_dataset, cfg_model)
    train_pipline.train_model()