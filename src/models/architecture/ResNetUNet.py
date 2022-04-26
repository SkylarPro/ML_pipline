#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
from torchvision import models

base_model = models.resnet18(pretrained=True)

def convrelu_duble(in_channels, out_channels, kernel, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, padding=padding),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel, padding=padding),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )

def convrelu_vanil(in_channels, out_channels, kernel, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, padding=padding),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )

def classifier_head(in_channels, number_class):
    return nn.Sequential(
                         nn.Linear(in_channels, in_channels//2),
                         nn.ReLu(),
                         nn.Linear(in_channels//2, number_class),
                         nn.Softmax(),
                        )
                       
class ResNetUNet(nn.Module):

    def __init__(self, n_class, n_channels):
        super(ResNetUNet,self).__init__()
        self.name = "ResNetUnet"
        self.base_layers = list(base_model.children())      
        
        self.layer0 = nn.Sequential(*self.base_layers[:3]) # size=(N, 64, x.H/2, x.W/2)
        self.layer0_1x1 = convrelu_vanil(64, 64, 1, 0)
        self.layer1 = nn.Sequential(*self.base_layers[3:5]) # size=(N, 64, x.H/4, x.W/4)        
        self.layer1_1x1 = convrelu_vanil(64, 64, 1, 0)       
        self.layer2 = self.base_layers[5]  # size=(N, 128, x.H/8, x.W/8)        
        self.layer2_1x1 = convrelu_vanil(128, 128, 1, 0)  
        self.layer3 = self.base_layers[6]  # size=(N, 256, x.H/16, x.W/16)        
        self.layer3_1x1 = convrelu_vanil(256, 256, 1, 0)  
        self.AVGPooling = nn.AdaptiveAvgPool2d(output_size=(1,1))
        self.cls_head = nn.Linear(256, n_class)#classifier_head(256, 100)
                
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        self.conv_up2 = convrelu_duble(128 + 256, 256, 3, 1)
        self.conv_up1 = convrelu_duble(64 + 256, 256, 3, 1)
        self.conv_up0 = convrelu_duble(64 + 256, 128, 3, 1)
        
        self.conv_original_size0 = convrelu_vanil(3, 64, 3, 1)
        self.conv_original_size1 = convrelu_vanil(64, 64, 3, 1)
        self.conv_original_size2 = convrelu_vanil(64 + 128, 64, 3, 1)
        
        self.conv_last = nn.Conv2d(64, n_channels, 1)
        
        
    def forward(self, input):
        x_original = self.conv_original_size0(input)
        x_original = self.conv_original_size1(x_original)
        
        layer0 = self.layer0(input)            
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)  
        
        pool = self.AVGPooling(layer3).reshape(layer3.shape[0], -1)
        cls_out = self.cls_head(pool)
        
        layer3 = self.layer3_1x1(layer3)
        x = self.upsample(layer3)
        layer2 = self.layer2_1x1(layer2)
        x = torch.cat([x, layer2], dim=1) 
        x = self.conv_up2(x)

        x = self.upsample(x)
        layer1 = self.layer1_1x1(layer1)
        x = torch.cat([x, layer1], dim=1) 
        x = self.conv_up1(x) 

        x = self.upsample(x)
        layer0 = self.layer0_1x1(layer0)
        x = torch.cat([x, layer0], dim=1)
        x = self.conv_up0(x)
        
        x = self.upsample(x)
        x = torch.cat([x, x_original], dim=1)
        x = self.conv_original_size2(x)        
        
        out = self.conv_last(x)
        
        return out, cls_out 