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

DATASET_LOC = '/home/wluo/vidor-dataset'
TRAIN_LOG_LOC = 'output/train.log'

batch_size = 1
classes = 42


def sigmoid(x):
    return 1/(1+np.exp(-x))

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


# train the model
def run(model, dataloader, lr, num_epochs=50):

    optimizer = optim.Adam(model.parameters(), lr=lr)
    lr_sched = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, verbose=True)

    since = time.time()

    best_loss = 10000
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        probs = []
        train_step(model, optimizer, dataloader['train'])
        prob_val, val_loss = val_step(model, dataloader['val'])
        probs.append(prob_val)
        lr_sched.step(val_loss)

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), 'models/best.pt')

def eval_model(model, dataloader, baseline=False):
    results = {}
    print('evalating...')
    for data in tqdm(dataloader):
        other = data[3]
        outputs, loss, probs, _ = run_network(model, data, 0, baseline)
        fps = outputs.size()[1]/other[1][0]

        results[other[0][0]] = (outputs.data.cpu().numpy()[0], probs.data.cpu().numpy()[0], data[2].numpy()[0], fps)
    return results

def run_network(model, data, baseline=False):
    # get the inputs
    inputs, mask, labels, other = data
    
    # wrap them in Variable
    inputs = Variable(inputs.cuda())
    mask = Variable(mask.cuda())
    labels = Variable(labels.cuda())
    
    cls_wts = torch.FloatTensor([1.00]).cuda()

    # forward
    if not baseline:
        outputs = model([inputs, torch.sum(mask, 1)])
        outputs = outputs.permute(0,2,1)
    else:
        outputs = model(inputs)
        outputs = outputs.squeeze(3).squeeze(3).permute(0,2,1) # remove spatial dims
    probs = torch.sigmoid(outputs) * mask.unsqueeze(2)
    print(outputs.size())
    print(mask.size())
    # binary action-prediction loss
    loss = F.binary_cross_entropy_with_logits(outputs, labels)#, weight=cls_wts)
    # loss = torch.sum(loss) / torch.sum(mask) # mean over valid entries
    print(mask.data.cpu().numpy())
    print(loss.data.item())

    input()
    # compute accuracy
    corr = torch.sum(mask)
    tot = torch.sum(mask)

    return outputs, loss, probs, corr/tot
            
                

def train_step(model, optimizer, dataloader):
    model.train(True)
    tot_loss = 0.0
    error = 0.0
    num_iter = 0.
    print('training...')
    # Iterate over data.
    for data in tqdm(dataloader):
        num_iter += 1
        optimizer.zero_grad()
        
        outputs, loss, probs, err = run_network(model, data)
        error += err.data.item()
        print(err.data.item())
        tot_loss += loss.data.item()
        
        loss.backward()
        optimizer.step()
    optimizer.step()
    optimizer.zero_grad()
    epoch_loss = tot_loss / num_iter
    error = error / num_iter
    print('train-{} Loss: {:.4f} Acc: {:.4f}'.format(dataloader.root, epoch_loss, error))

  

def val_step(model, dataloader):
    model.train(False)
    apm = APMeter()
    tot_loss = 0.0
    error = 0.0
    num_iter = 0.
    num_preds = 0

    full_probs = {}


    # Iterate over data.
    for data in tqdm(dataloader):
        num_iter += 1
        other = data[3]
        
        outputs, loss, probs, err = run_network(model, data)
        apm.add(probs.data.cpu().numpy()[0], data[2].numpy()[0])
        
        error += err.data.item()
        tot_loss += loss.data.item()
        
        # post-process preds
        outputs = outputs.squeeze()
        probs = probs.squeeze()
        fps = outputs.size()[1]/other[1][0]
        full_probs[other[0][0]] = (probs.data.cpu().numpy().T, fps)
        
        
    epoch_loss = tot_loss / num_iter
    error = error / num_iter
    print('val-map:', apm.value().mean())
    apm.reset()
    print('val-{} Loss: {:.4f} Acc: {:.4f}'.format(dataloader.root, epoch_loss, error))
    
    return full_probs, epoch_loss


model_f = models.get_hier

if __name__ == '__main__':

    
    dataloaders, datasets = load_data()


    if args.train:
        model = nn.DataParallel(model_f(classes))
    
        lr = 0.1*batch_size/len(datasets['train'])    
        run(model, dataloaders, lr, num_epochs=60)

    else:
        print('Evaluating...')
        rgb_model = nn.DataParallel(torch.load('models/best.pt'))
        rgb_model.train(False)
        
        dataloaders, datasets = load_data('', test_split, flow_root)

        rgb_results = eval_model(rgb_model, dataloaders['val'])

        # flow_model = nn.DataParallel(torch.load(args.flow_model_files))
        # flow_model.train(False)
            
        # dataloaders, datasets = load_data('', test_split, flow_root)
            
        # flow_results = eval_model(flow_model, dataloaders['val'])

        apm = APMeter()


        for vid in rgb_results.keys():
            o,p,l,fps = rgb_results[vid]
         
            if vid in flow_results:
                o2,p2,l2,fps = flow_results[vid]
                o = (o[:o2.shape[0]]*.5+o2*.5)
                p = (p[:p2.shape[0]]*.5+p2*.5)
            apm.add(sigmoid(o), l)
        print('MAP:', apm.value().mean())
