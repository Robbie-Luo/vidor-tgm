from __future__ import division
import argparse
import os
import logging
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torchvision
from torchvision import datasets, transforms
import models
from apmeter import APMeter
from vidor_i3d_per_video import Vidor as Dataset
from vidor_i3d_per_video import mt_collate_fn as collate_fn
DATASET_LOC = '/home/shuogpu/wluo/vidor-dataset'
TRAIN_LOG_LOC = 'output/train.log'

batch_size = 1

def load_data(num_workers = 4, root = 'vidor'):
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
    lr = 0.1*batch_size/len(dataset_train)    
    return dataloaders, datasets, lr

def print_log(line):
    logging.info(line)
    print(line)

def print_statement(statement, isCenter=False, symbol='=', number=15, newline=False, verbose=None):
    '''
    Print required statement in a given format.
    '''
    if verbose is not None:
        if verbose > 0:
            if newline:
                print()
            if number > 0:
                prefix = symbol * number + ' '
                suffix = ' ' + symbol * number
                statement = prefix + statement + suffix
            if isCenter:
                print(statement.center(os.get_terminal_size().columns))
            else:
                print(statement)
        else:
            pass
    else:
        if newline:
            print()
        if number > 0:
            prefix = symbol * number + ' '
            suffix = ' ' + symbol * number
            statement = prefix + statement + suffix
        if isCenter:
            print(statement.center(os.get_terminal_size().columns))
        else:
            print(statement)

def run(dataloaders, lr , save_model='output/best.pt', num_epochs = 60, num_classes=42):
    model = nn.DataParallel(models.get_hier(num_classes))
    optimizer = optim.Adam(model.parameters(), lr=lr)
    lr_sched = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, verbose=True)
    max_iterations = len(dataloaders['train'])    
    best_loss = 100
    for epoch in range(num_epochs):
        # Training phase
        print_statement('MODEL TRAINING')
        total_loss = 0
        iteration = 0
        for data in dataloaders['train']:
            iteration += 1
            # get the inputs
            inputs, mask, labels, other = data

            # wrap them in Variable
            inputs = Variable(inputs.cuda())
            mask = Variable(mask.cuda())
            labels = Variable(labels.cuda())
          
            # set the model to train
            model.train(True)
            optimizer.zero_grad()
            # get the outputs
            outputs = model([inputs,torch.sum(mask, 1)])
            outputs = outputs.permute(0,2,1)
            loss = F.binary_cross_entropy_with_logits(outputs, labels)
            total_loss += loss
            loss.backward()
            optimizer.step()
            if iteration % 10 == 0:
                print_log('Epoch:{:d}/{:d},Train step:{:d}/{:d},Loss:{:4f}'.format(epoch+1, num_epochs, iteration, max_iterations, loss))
        print_log('Epoch:{:d}/{:d}, total_loss:{:4f}'.format(epoch+1,num_epochs,total_loss/max_iterations))
        # Validation phase
        print_statement('MODEL VALIDATING')
        # set the model to evaluate
        model.train(False)
        optimizer.zero_grad()
        tot_loss = 0.0  
        # Iterate over data.
        iteration = 0
        for data in tqdm(dataloaders['val']):
            iteration += 1
            # get the inputs
            inputs, mask, labels, other = data
            # wrap them in Variable
            inputs = Variable(inputs.cuda())
            mask = Variable(mask.cuda())
            labels = Variable(labels.cuda())
            
            outputs = model([inputs,torch.sum(mask, 1)])
            outputs = outputs.permute(0,2,1)
            loss = F.binary_cross_entropy_with_logits(outputs, labels)
            tot_loss += loss.data.item()
        val_loss = total_loss/iteration
        print_log(('Valadation: Loss:{:4f}'.format(val_loss))
        lr_sched.step(val_loss)
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), 'models/best.pt')
    

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, filename=TRAIN_LOG_LOC, filemode="w+",
                        format="%(asctime)-15s %(levelname)-8s %(message)s")
    dataloaders, datasetsm, lr = load_data()
    run(dataloaders, lr)