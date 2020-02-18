from __future__ import division
import time
import os
import argparse
from tqdm import tqdm
def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser()
parser.add_argument('-mode', type=str, default = 'rgb', help='rgb or flow (or joint for eval)')
parser.add_argument('-train', type=str2bool, default='True', help='train or eval')

args = parser.parse_args()


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable

import torchvision
from torchvision import datasets, transforms


import numpy as np
import json

import models
from apmeter import APMeter
from vidor_i3d_per_video import Vidor as Dataset
from vidor_i3d_per_video import mt_collate_fn as collate_fn

DATASET_LOC = '/home/shuogpu/wluo/vidor-dataset'
TRAIN_LOG_LOC = 'output/train.log'

batch_size = 10

def load_data(num_workers = 16, root = 'vidor'):
    # Load Data

    dataset_train = Dataset(DATASET_LOC, 'training')
    dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=num_workers,collate_fn=collate_fn,
                                                pin_memory=True)
    dataloader_train.root = root
    dataset_val = Dataset(DATASET_LOC, 'validation')
    dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=1, shuffle=True, num_workers=num_workers, collate_fn=collate_fn,
                                                pin_memory=True)
    dataloader_val.root = root                                            
    dataloaders = {'train': dataloader_train, 'val': dataloader_val}
    datasets = {'train': dataset_train, 'val': dataset_val}
    return dataloaders, datasets

dataloaders, datasets = load_data()

for data in tqdm(dataloaders['train']):
    