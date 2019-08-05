import numpy as np
from functools import partial
import pandas as pd
import os
from tqdm import tqdm_notebook, tnrange, tqdm
import sys
import torch
from torch import nn
from torch.nn.init import kaiming_normal
import torch.nn.functional as F
from torch.optim import SGD,Adam
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torch.optim.optimizer import Optimizer
import torchvision
from torchvision import models
import pretrainedmodels
from pretrainedmodels.models import *
from torch import nn
from config import config
from collections import OrderedDict
import torch.nn.functional as F
from torchvision import transforms as T
from imgaug import augmenters as iaa
import random
import pathlib
import cv2
import json
from PIL import Image
import torch
from torchvision import transforms

from efficientnet_pytorch import EfficientNet

# create dataset class
class MultiModalDataset(Dataset):
    def __init__(self,images_df, base_path,augument=True,mode="train"):
        if not isinstance(base_path, pathlib.Path):
            base_path = pathlib.Path(base_path)
        self.images_df = images_df.copy() #csv
        self.augument = augument
        self.images_df.Id = self.images_df.Id.apply(lambda x:base_path / str(x).zfill(6))
        self.mode = mode

    def __len__(self):
        return len(self.images_df)

    def __getitem__(self,index):
        X = self.read_images(index)
        if not self.mode == "test":
            y = self.images_df.iloc[index].Target
        else:
            y = str(self.images_df.iloc[index].Id.absolute())
        if self.augument:
            X = self.augumentor(X)
        X = T.Compose([T.ToPILImage(),T.ToTensor()])(X)
        return X.float(),y

    def read_images(self,index):
        row = self.images_df.iloc[index]
        filename = str(row.Id.absolute())
        images = cv2.imread(filename+'.jpg')
        return images

    def augumentor(self,image):
        augment_img = iaa.Sequential([
            iaa.Fliplr(0.5),
            iaa.Flipud(0.5),
            iaa.SomeOf((0,4),[
                iaa.Affine(rotate=90),
                iaa.Affine(rotate=180),
                iaa.Affine(rotate=270),
                iaa.Affine(shear=(-16, 16)),
            ]),
            iaa.OneOf([
                    iaa.GaussianBlur((0, 3.0)), # blur images with a sigma between 0 and 3.0
                    iaa.AverageBlur(k=(2, 7)), # blur image using local means with kernel sizes between 2 and 7
                    iaa.MedianBlur(k=(3, 11)), # blur image using local medians with kernel sizes between 2 and 7
                ]),
            #iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)), # sharpen images
            ], random_order=True)
        image_aug = augment_img.augment_image(image)
        return image_aug


class _LRScheduler(object):
    def __init__(self, optimizer, last_epoch=-1):
        if not isinstance(optimizer, Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))
        self.optimizer = optimizer
        if last_epoch == -1:
            for group in optimizer.param_groups:
                group.setdefault('initial_lr', group['lr'])
        else:
            for i, group in enumerate(optimizer.param_groups):
                if 'initial_lr' not in group:
                    raise KeyError("param 'initial_lr' is not specified "
                                   "in param_groups[{}] when resuming an optimizer".format(i))
        self.base_lrs = list(map(lambda group: group['initial_lr'], optimizer.param_groups))
        self.step(last_epoch + 1)
        self.last_epoch = last_epoch

    def get_lr(self):
        raise NotImplementedError

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr

class CosineAnnealingLR(_LRScheduler):
    def __init__(self, optimizer, T_max, eta_min=0, last_epoch=-1):
        self.T_max = T_max
        self.eta_min = eta_min
        self.optimizer = optimizer
        super(CosineAnnealingLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [self.eta_min + (base_lr - self.eta_min) *
                (1 + np.cos(np.pi * self.last_epoch / self.T_max)) / 2
                for base_lr in self.base_lrs]
    
    def _reset(self, epoch, T_max):
        """
        Resets cycle iterations.
        Optional boundary/step size adjustment.
        """
        return CosineAnnealingLR(self.optimizer, self.T_max, self.eta_min, last_epoch=epoch)


class FCViewer(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class MultiModalNet(nn.Module):
    def __init__(self, backbone1, drop):
        super(MultiModalNet,self).__init__()
        self.img_model = EfficientNet.from_pretrained('efficientnet-b5',num_classes=9)
        img_model = pretrainedmodels.__dict__[backbone1](num_classes=1000, pretrained='imagenet') #seresnext101
        img_model2 = pretrainedmodels.__dict__['seresnet152'](num_classes=1000, pretrained='imagenet') #seresnext101

        self.img_encoder = list(img_model.children())[:-2]
        self.img_encoder.append(nn.AdaptiveAvgPool2d(1))
        self.img_encoder = nn.Sequential(*self.img_encoder)

        self.img_encoder2 = list(img_model2.children())[:-2]
        self.img_encoder2.append(nn.AdaptiveAvgPool2d(1))
        self.img_encoder2 = nn.Sequential(*self.img_encoder)


        if drop > 0:
            self.img_fc = nn.Sequential(FCViewer(),
                                    nn.Dropout(drop),
                                    nn.Linear(img_model.last_linear.in_features, 128))
            self.img_fc3 = nn.Sequential(FCViewer(),
                                    nn.Dropout(drop),
                                    nn.Linear(img_model2.last_linear.in_features, 128))
        self.img_fc2 = nn.Linear(2048, 128)

        self.cls = nn.Linear(384,config.num_classes)

    def forward(self, x_img):
        x_img1 = self.img_model(x_img)
        x_img1 = self.img_fc2(x_img1)
        x_img2 = self.img_encoder(x_img)
        x_img2 = self.img_fc(x_img2)
        x_img3 = self.img_encoder2(x_img)
        x_img3 = self.img_fc3(x_img3)
        x_img = torch.cat((x_img1,x_img2,x_img3),1)
        x_cat = self.cls(x_img)
        return x_cat
