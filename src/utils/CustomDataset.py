from torch.utils.data import Dataset
from torch.utils.data import SubsetRandomSampler
from src.utils.mixin.config import Args
from torchvision.datasets import CIFAR100
import numpy as np
import torch
from torchvision import transforms
from typing import Dict


class DatasetCIFAR100(Dataset, Args):
    
    def __init__(self, cfg_dataset:Dict):
        self.args = Args(cfg_dataset)
        
        self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                ])
        self.train_ds, self.val_ds = self.sempler(CIFAR100(self.args.path_to_data,
                                                     transform = self.transform,
                                                     train=True),
                                            
                                            batch_size = self.args.bs,
                                            split = self.args.val_size,
                                           )
        self.test_ds = torch.utils.data.DataLoader(CIFAR100(self.args.path_to_data,
                                transform = self.transform,
                                train=False),
                                
                                            batch_size = self.args.bs
                                                  )
        self.visual_ds = None
        if hasattr(self.args,"visual_val") and self.args.visual_val:
            self.visual_ds = torch.utils.data.DataLoader(CIFAR100(self.args.path_to_data,
                                transform = self.transform,
                                train=False),
                                            batch_size = 1,
                                            shuffle=False,
                                            sampler=SubsetRandomSampler([*range(10)])
                                            )
    
    
    def post_proc(self,img):
        """
        Restoring the image after passing through the model
        """
        invTrans = transforms.Compose([ 
                                lambda x:x.cpu().detach(),
                                transforms.Normalize(mean = [ 0., 0., 0. ],
                                                     std = [ 1/0.229, 1/0.224, 1/0.225 ]),
                                transforms.Normalize(mean = [ -0.485, -0.456, -0.406 ],
                                                     std = [ 1., 1., 1. ]),
                                lambda x:x*((2**8) - 1),
                                lambda x:x.permute((0,3,2,1)),
                                lambda x:x.type(torch.ByteTensor).numpy()[0]
                               ])
        return invTrans(img)
        
    def sempler(self, data_train, batch_size = 4, split = .2):
        """
        Splitting a dataset into two with certain proportions
        """
        data_size = len(data_train)
        
        validation_split = split
        split = int(np.floor(validation_split * data_size))
        indices = list(range(data_size))
        np.random.shuffle(indices)
    
        train_indices, val_indices = indices[split:], indices[:split]
    
        train_sampler = SubsetRandomSampler(train_indices)
        val_sampler = SubsetRandomSampler(val_indices)
        
    
        train_loader = torch.utils.data.DataLoader(data_train, batch_size=batch_size,
                                                  sampler=train_sampler,)
        val_loader = torch.utils.data.DataLoader(data_train, batch_size=batch_size,
                                                sampler=val_sampler,)
    
        return train_loader, val_loader
    
    @property
    def train_loader(self,):
        return self.train_ds
    @property
    def val_loader(self,):
        return self.val_ds
    @property
    def test_loader(self,):
        return self.test_ds
    @property
    def visual_loader(self,):
        return self.visual_ds
    @property
    def classes_test(self,):
        return self.test_ds.dataset.classes
