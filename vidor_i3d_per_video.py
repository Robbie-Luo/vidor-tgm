import torch
import torch.utils.data as data_utl
from torch.utils.data.dataloader import default_collate

import numpy as np
import json
import csv
import h5py
from tqdm import tqdm
import os
import os.path
from dataset import VidOR
import pickle
def video_to_tensor(pic):
    """Convert a ``numpy.ndarray`` to tensor.
    Converts a numpy.ndarray (T x H x W x C)
    to a torch.FloatTensor of shape (C x T x H x W)
    
    Args:
         pic (numpy.ndarray): Video to be converted to tensor.
    Returns:
         Tensor: Converted video.
    """
    return torch.from_numpy(pic.transpose([3,0,1,2]))

def load_vidor_dataset():
    dataset_path ='/home/wluo/vidor-dataset'
    anno_path = os.path.join(dataset_path,'annotation')
    video_path = os.path.join(dataset_path,'video')
    frame_path = os.path.join(dataset_path,'frame')
    if not os.path.exists('dataset/vidor-dataset.pkl'):
        dataset = VidOR(anno_path, video_path, ['training', 'validation'], low_memory=True)
        with open('dataset/vidor-dataset.pkl','wb') as file:
            pickle.dump(dataset,file)
    else:
        with open('dataset/vidor-dataset.pkl','rb') as file:
            dataset = pickle.load(file)
    return dataset

# make_dataset('multithumos.json', 'training', '/ssd2/thumos/val_i3d_rgb')

class Vidor(data_utl.Dataset):

    def __init__(self, root, split):
        self.root = root
        self.split = split
        self.num_classes = 42
        self.data = self.make_vidor_data()
        
    def make_vidor_data(self):
        pkl_path = f'dataset/vidor_{self.split}_feature_data.pkl'
        if not os.path.exists(pkl_path):
            vidor_dataset = load_vidor_dataset()
            vidor_data = []
            actions = vidor_dataset._get_action_predicates()
            vids = vidor_dataset.get_index(self.split)
            for ind in tqdm(vids):
                video_path = vidor_dataset.get_video_path(ind,self.split)
                frame_dir = video_path.replace('video','frame').replace('.mp4','')
                num_frames = len(os.listdir(frame_dir))
                feature_dir = video_path.replace('video','feature').replace('.mp4','')
                feat = np.load(os.path.join(feature_dir,'i3d_040.npy'))
                num_feat = feat.shape[0]
                label = np.zeros((num_feat,self.num_classes), np.float32)
                fps = num_feat/num_frames
                for each_ins in vidor_dataset.get_action_insts(ind):
                    start_f, end_f = each_ins['duration']
                    action = actions.index(each_ins['category'])
                    for fr in range(0,num_feat,1):
                        if fr/fps >= start_f and fr/fps <= end_f:
                            label[fr, action-1] = 1 # binary classification
                vidor_data.append((feature_dir,label,num_frames))
            with open(pkl_path,'wb') as file:
                pickle.dump(vidor_data,file)
        else:
            with open(pkl_path,'rb') as file:
                vidor_data = pickle.load(file)
        return vidor_data

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        feature_dir,label,num_frames = self.data[index]
        feature_path = os.path.join(feature_dir.replace('/home/wluo/vidor-dataset',self.root),'i3d_040.npy')
        if os.path.exists(feature_path):
            feat = np.load(feature_path)
            feat = feat.reshape((feat.shape[0],1,1,1024))
            #r = np.random.randint(0,10)
            #feat = feat[:,r].reshape((feat.shape[0],1,1,1024))
            feat = feat.astype(np.float32)
            #self.in_mem[entry[0]] = feat
        else:
            print(feature_path)
        return feat,label, [index, num_frames]

    def __len__(self):
        return len(self.data)


    
def mt_collate_fn(batch):
    "Pads data and puts it into a tensor of same dimensions"
    max_len = 0
    for b in batch:
        if b[0].shape[0] > max_len:
            max_len = b[0].shape[0]

    new_batch = []
    for b in batch:
        f = np.zeros((max_len, b[0].shape[1], b[0].shape[2], b[0].shape[3]), np.float32)
        m = np.zeros((max_len), np.float32)
        l = np.zeros((max_len, b[1].shape[1]), np.float32)
        f[:b[0].shape[0]] = b[0]
        m[:b[0].shape[0]] = 1
        l[:b[0].shape[0], :] = b[1]
        new_batch.append([video_to_tensor(f), torch.from_numpy(m), torch.from_numpy(l), b[2]])

    return default_collate(new_batch)
    
