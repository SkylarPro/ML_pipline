#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import sys
import os
from functools import partial
import torch
from tqdm import tqdm

sys.path.append(os.getcwd())
from src.utils.CustomDataset import DatasetCIFAR100
from src.utils.criterions import (L1, L2, CrossEntropy,Accuracy,SSIM,Accum_prediction, Accum_gt,
                                  compute_criterion)
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics

from src.models.model_loader import load_model
from src.utils.logs import AccumulativeDict,plot_params,res_to_string
from src.utils.mixin.config import Args


config_inf = {
    "log_dir": "inference/logs",
}
cfg_model = {
   "name":"AutoEncoder",
   "count_cls":100,
   "number_channel": 3,
"weights_path":"src/pipeline/logs/AutoEncoder_mode_img_to_img_epoch_60/weight/AutoEncoder_mode_img_to_cls_epoch_123.pt"
}
cfg_dataset = {
    "path_to_data": "data",
    "bs": 16,
    "val_size": 0.2
}
class Inference:
    
    def __init__(self,config_inf = None, config_data = None, config_model = None):
        config_inf.update(config_model)
        self.args = Args(config_inf)
        self.cfg_model = Args(config_model)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.test_ds = DatasetCIFAR100(config_data)
        self.model = load_model(self.cfg_model)
        self._set_path()
        
        if not os.path.isdir(self.args.save_log):
            os.makedirs(self.args.save_log)
        
        self.define_metric()
        self.compute_criterion_cls = partial(compute_criterion,
                                         loss_list = None,
                                         metric_list = self.metrics_cls
                                         )
    def _set_path(self,):
        setattr(self.args,"save_log",os.path.join(self.args.log_dir, 
                                                   self.model.name_chpt))           
    def define_metric(self,):
        self.metrics_img = [L1(metric = True), L2(metric = True), SSIM(metric = True)]
        self.metrics_cls = [Accuracy(), Accum_prediction(), Accum_gt()]
        
        
    def plot_confusion_matrix(self, pred, gt):
        matrix_cf = metrics.confusion_matrix(gt, pred)
        df_cm = pd.DataFrame(matrix_cf, index = [name for name in self.test_ds.classes_test],
                          columns = [name for name in self.test_ds.classes_test])
        plt.figure(figsize = (50,50))
        sn.heatmap(df_cm, annot=True)
        plt.savefig(f"{self.args.save_log}/confmatrix.png")
        print("Create_confusion")
        
        
    def apply(self,):
        apply_log = AccumulativeDict()
        self.model.to(self.device)
        with torch.no_grad():
            for i_step, data  in tqdm(enumerate(self.test_ds.test_loader)):
                
                imgs_gpu = data[0].to(self.device)
                labels_gpu = data[1].to(self.device)
    
                prediction_img, prediction_cls = self.model(imgs_gpu)
                
                _, _, metric_log = self.compute_criterion_cls(
                                                    prediction_cls, labels_gpu,
                                                    )
                apply_log += metric_log
                                        
                del imgs_gpu
                del labels_gpu
        self.plot_confusion_matrix(apply_log["Accum_pred"], apply_log["Accum_gt"],)
        del apply_log["Accum_pred"]
        del apply_log["Accum_gt"]
        apply_log /= (i_step + 1)   
        print(res_to_string(apply_log,0,0))
        return apply_log
    
    
if __name__=="__main__":
    infernce = Inference(config_inf, cfg_dataset, cfg_model)
    infernce.apply()

