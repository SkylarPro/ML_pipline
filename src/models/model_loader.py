#!/usr/bin/env python
# coding: utf-8

import torch
from .ResNetUNet import ResNetUNet

def load_model(cfg):
    if cfg.name == "ResNetUnet": 
        model = ResNetUNet(cfg.count_cls, cfg.number_channel)
        if hasattr(cfg, "weights_path"):
            print("Load weight in model...")
            model.load_state_dict(torch.load(cfg.weights_path))
    return model