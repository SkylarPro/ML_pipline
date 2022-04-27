#!/usr/bin/env python
# coding: utf-8

# In[2]:


import sys
import os
import argparse
from functools import partial

sys.path.append(os.getcwd())

from src.utils.mixin.config import Args
from src.utils.CustomDataset import DatasetCIFAR100
from src.utils.logs import AccumulativeDict, plot_params, res_to_string
from src.utils.criterions import (L1, L2, CrossEntropy, Accuracy, SSIM,
                                  compute_criterion)
from src.models.model_loader import load_model

import torch.nn as nn
import numpy as np
import torch
from torch.optim import Adam, lr_scheduler
from torch.utils.tensorboard import SummaryWriter
import cv2

from typing import Dict
from tqdm import tqdm

# mode img_to_img, img_to_cls, combinate
config_train = {
    "mode": "img_to_cls",
    "visual_val": True,
    "lr": 3e-04,
    "weight_decay": 1e-05,
    "log_dir": "src/pipeline/logs",
    "T_0": 10,
    "T_mult": 2,
    "eta_min": 1e-09,
    "n_epoch": 150,
    "bs": 64,
    "p_save_model": "weight"
}

cfg_dataset = {
    "path_to_data": "data",
    "bs": 64,
    "val_size": 0.2
}
# models name AutoEncoder, ResNetUnet
cfg_model = {
    "name": "AutoEncoder",
    "count_cls": 100,
    "number_channel": 3,
    "weights_path": "src/pipeline/logs/AutoEncoder_first_train/weight/AutoEncoder_mode_img_to_img_epoch_60.pt"
}


class Pipline:
    def __init__(self, config_train: Dict, config_data: Dict, config_model: Dict):
        config_train, config_data, config_model = self._merge_config(config_train,
                                                                     config_data, config_model)
        if config_model.get("weights_path") != None:
            self.epoch_restore = int(config_model.get("weights_path")[:-3].split("_")[-1])
        else:
            self.epoch_restore = 0
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.args = Args(config_train)
        self.cfg_model = Args(config_model)

        self.dataset = DatasetCIFAR100(config_data)
        self.model = load_model(self.cfg_model)

        self._set_path()
        if not os.path.isdir(self.args.log_dir):
            os.makedirs(self.args.save_ckpt)
            os.makedirs(self.args.train_path)
            os.makedirs(self.args.val_path)

        if not os.path.isdir(self.args.visual_path):
            os.makedirs(self.args.visual_path)
        if not os.path.isdir(self.args.save_ckpt):
            os.makedirs(self.args.save_ckpt)

        self.define_loss()
        self.define_metric()

        self.optimizer = Adam([param for param in self.model.parameters() if param.requires_grad],
                              lr=self.args.lr, weight_decay=self.args.weight_decay
                              )
        self.scheduler = lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer,
                                                                  T_0=self.args.T_0,
                                                                  T_mult=self.args.T_mult,
                                                                  eta_min=self.args.eta_min)

        self.compute_criterion_img = partial(compute_criterion,
                                             loss_list=self.losses_img,
                                             metric_list=self.metrics_img
                                             )
        self.compute_criterion_cls = partial(compute_criterion,
                                             loss_list=self.losses_cls,
                                             metric_list=self.metrics_cls
                                             )

        if self.args.log_dir:
            self.writer_train = SummaryWriter(self.args.train_path)
            self.writer_val = SummaryWriter(self.args.val_path)

    def _set_path(self, ):
        setattr(self.args, "save_ckpt", os.path.join(self.args.log_dir, self.model.name_chpt, self.args.p_save_model))
        setattr(self.args, "train_path", os.path.join(self.args.log_dir, self.model.name_chpt, "train"))
        setattr(self.args, "val_path", os.path.join(self.args.log_dir, self.model.name_chpt, "val"))
        setattr(self.args, "visual_path", os.path.join(self.args.log_dir, self.model.name_chpt, "visual",
                                                       ))

    def _merge_config(self, cfg_train, cfg_data, cfg_model):
        """
        Merging some entities from configs
        """
        cfg_train.update({"name": cfg_model["name"]})
        cfg_model.update({"mode": cfg_train["mode"]})
        cfg_data.update({"visual_val": cfg_train["visual_val"]})
        return cfg_train, cfg_data, cfg_model

    def define_loss(self, ):
        if self.args.mode == "img_to_img":
            self.losses_img = [
                L1(weight=0.15),
                SSIM(weight=0.85),
            ]
            self.losses_cls = None
        elif self.args.mode == "img_to_cls":
            self.losses_img = None
            self.losses_cls = [
                CrossEntropy(weight=1.0)
            ]
        elif self.args.mode == "combinate":
            self.losses_img = [
                L1(weight=1.0),
                SSIM(weight=1.0),
            ]
            self.losses_cls = [
                CrossEntropy(weight=1.0)
            ]
        else:
            assert False, f"Mode should img_to_img, img_to_cls or combinate not {self.args.mode}"

    def define_metric(self, ):
        self.metrics_img = [L2(weight=1.0, metric=True)]
        self.metrics_cls = [Accuracy()]

    def train_model(self, ):
        self.model.to(self.device)
        for epoch in range(self.args.n_epoch):

            train_log = AccumulativeDict()
            self.model.train()
            for i_step, data in tqdm(enumerate(self.dataset.train_loader)):
                imgs_gpu = data[0].to(self.device)
                labels_gpu = data[1].to(self.device)

                prediction_img, prediction_cls = self.model(imgs_gpu)

                total_loss_img, loss_log, metric_log = self.compute_criterion_img(
                    prediction_img.cpu(), imgs_gpu.cpu(),
                )
                train_log += loss_log
                train_log += metric_log
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
            print(f"Save model to {self.args.save_ckpt}")
            torch.save(self.model.state_dict(),
                       f"{self.args.save_ckpt}/{self.model.name}_mode_{self.args.mode}_epoch_{epoch + self.epoch_restore}.pt")

            print("Train: ", res_to_string(train_log, epoch + self.epoch_restore, self.args.n_epoch))

            val_log = self.compute_valid()

            print("Val: ", res_to_string(val_log, epoch + self.epoch_restore, self.args.n_epoch))

            self.scheduler.step(epoch)

            plot_params(train_log, self.writer_train, epoch)
            plot_params(val_log, self.writer_val, epoch)

        self.writer_val.close()
        self.writer_train.close()
        return self.model

    def compute_visual(self, ):
        """
        Visual validation
        """
        self.model.to(self.device)
        self.model.eval()
        with torch.no_grad():
            for i_step, data in tqdm(enumerate(self.dataset.visual_loader)):
                imgs_gpu = data[0].to(self.device)
                prediction_img, _ = self.model(imgs_gpu)
                img_pred = self.dataset.post_proc(prediction_img)
                img_gt = self.dataset.post_proc(imgs_gpu)
                cv2.imwrite(f"{self.args.visual_path}/img_{i_step}_predict.png", img_pred)
                cv2.imwrite(f"{self.args.visual_path}/img_{i_step}_gt.png", img_gt)
        return True

    def compute_valid(self, ):
        self.model.to(self.device)
        self.model.eval()

        val_log = AccumulativeDict()
        with torch.no_grad():
            for i_step, data in tqdm(enumerate(self.dataset.val_loader)):
                imgs_gpu = data[0].to(self.device)
                labels_gpu = data[1].to(self.device)

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Inforamtion about learning')
    parser.add_argument('--stage', type=str, default="train", help='Stage can be [train, valid, compute_visual]')
    args = parser.parse_args()
    if args.stage == "train":
        train_pipline = Pipline(config_train, cfg_dataset, cfg_model)
        train_pipline.train_model()
    elif args.stage == "valid":
        valid_pipline = Pipline(config_train, cfg_dataset, cfg_model)
        valid_pipline.compute_valid()
    elif args.stage == "compute_visual":
        visual_pipline = Pipline(config_train, cfg_dataset, cfg_model)
        visual_pipline.compute_visual()
    else:
        raise NotImplementedError(f"Now impl train,valid,compute_visual NOT {args.stage}")
