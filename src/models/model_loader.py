#!/usr/bin/env python
# coding: utf-8

import torch
from .architecture.models import ResNetUNet, AutoEncoder


def load_model(cfg):
    if cfg.name == "ResNetUnet":
        model = ResNetUNet(cfg)
        if hasattr(cfg, "weights_path"):
            print("Load weight in model...")
            model.load_state_dict(torch.load(cfg.weights_path))
        if hasattr(cfg, "mode") and cfg.mode == "img_to_cls":
            for name, param in model.named_parameters():
                if name.find("cls_head") == -1:
                    param.requires_grad = False
                else:
                    param.requires_grad = True
    elif cfg.name == "AutoEncoder":
        model = AutoEncoder(cfg)
        if hasattr(cfg, "weights_path"):
            print("Load weight in model...")
            model.load_state_dict(torch.load(cfg.weights_path))
        if hasattr(cfg, "mode") and cfg.mode == "img_to_cls":
            for name, param in model.named_parameters():
                if name.find("cls_head") == -1:
                    param.requires_grad = False
                else:
                    print(name)
                    param.requires_grad = True
    else:
        raise NotImplementedError
    return model
