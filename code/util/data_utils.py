# --------------------------------------------------------
# Based on BEiT and MAE code bases
# Pretrain and Finetune datasets for FL SSL.
# https://github.com/microsoft/unilm/tree/master/beit
# https://github.com/facebookresearch/mae
# Author: Rui Yan
# --------------------------------------------------------

import numpy as np
import pandas as pd

import os
from .datasets import DataAugmentationForPretrain, build_transform

from PIL import Image
from skimage.transform import resize
import cv2
import torch.utils.data as data

class DatasetFLPretrain(data.Dataset):
    """ data loader for pre-training """
    def __init__(self, args):    
                
        if args.split_type == 'central':
            cur_clint_path = os.path.join(args.data_path, args.split_type, args.single_client)
        else:
            cur_clint_path = os.path.join(args.data_path, f'{args.n_clients}_clients', 
                                            args.split_type, args.single_client)

        self.img_paths = list({line.strip().split(',')[0] for line in open(cur_clint_path)})
        
        self.labels = {line.strip().split(',')[0]: float(line.strip().split(',')[1]) for line in
                        open(os.path.join(args.data_path, 'labels.csv'))}
    
        self.transform = DataAugmentationForPretrain(args)
        self.args = args
    
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        index = index % len(self.img_paths)
        
        path = os.path.join(self.args.data_path, 'train', self.img_paths[index])
        name = self.img_paths[index]

        target = self.labels[name]
        target = np.asarray(target).astype('int64')
        
        if self.args.data_set == 'Retina':
            img = np.load(path)
            img = resize(img, (256, 256))
        else:
            img = np.array(Image.open(path).convert("RGB"))
        
        if img.ndim < 3:
            img = np.stack((img,)*3, axis=-1)
        elif img.shape[2] >= 3:
            img = img[:,:,:3]
        
        if self.transform is not None:
            img = Image.fromarray(np.uint8(img))
            sample = self.transform(img)
            
        return sample, target

    def __len__(self):
        return len(self.img_paths)


class DatasetFLFinetune(data.Dataset):
    """ data loader for fine-tuning """
    def __init__(self, args, phase, mode='finetune'):
        super(DatasetFLFinetune, self).__init__()
        self.phase = phase
        is_train = (phase == 'train')
        
        if not is_train: 
            args.single_client = os.path.join(args.data_path, f'{self.phase}.csv')
        
        if args.split_type == 'central':
            cur_clint_path = os.path.join(args.data_path, args.split_type, args.single_client)
        else:
            cur_clint_path = os.path.join(args.data_path, f'{args.n_clients}_clients', 
                                            args.split_type, args.single_client)
        
        self.img_paths = list({line.strip().split(',')[0] for line in open(cur_clint_path)})
        
        self.labels = {line.strip().split(',')[0]: float(line.strip().split(',')[1]) for line in
                        open(os.path.join(args.data_path, 'labels.csv'))}
        
        self.transform = build_transform(is_train, mode, args)
        
        self.args = args
    
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        index = index % len(self.img_paths)
        
        path = os.path.join(self.args.data_path, self.phase, self.img_paths[index])
        name = self.img_paths[index]
        
        try:
            target = self.labels[name]
            target = np.asarray(target).astype('int64')
        except:
            print(name, index)
        
        if self.args.data_set == 'Retina':
            img = np.load(path)
        else:
            img = np.array(Image.open(path).convert("RGB"))
        
        img = resize(img, (256, 256))
        
        if img.ndim < 3:
            img = np.concatenate((img,)*3, axis=-1)
        elif img.shape[2] >= 3:
            img = img[:,:,:3]
        
        if self.transform is not None:
            img = Image.fromarray(np.uint8(img))
            sample = self.transform(img)

        return sample, target

    def __len__(self):
        return len(self.img_paths)


def create_dataset_and_evalmetrix(args, mode='pretrain'):

    ## get the joined clients
    if args.split_type == 'central':
        args.dis_cvs_files = ['central']

    if args.split_type == 'central':
        args.dis_cvs_files = os.listdir(os.path.join(args.data_path, args.split_type))
    else:
        args.dis_cvs_files = os.listdir(os.path.join(args.data_path, f'{args.n_clients}_clients', args.split_type))
    
    args.clients_with_len = {}
    
    for single_client in args.dis_cvs_files:
        if args.split_type == 'central':
            img_paths = list({line.strip().split(',')[0] for line in
                            open(os.path.join(args.data_path, args.split_type, single_client))})
        else:
            img_paths = list({line.strip().split(',')[0] for line in
                                open(os.path.join(args.data_path, f'{args.n_clients}_clients',
                                                args.split_type, single_client))})
        args.clients_with_len[single_client] = len(img_paths)
    
    
    ## step 2: get the evaluation matrix
    args.learning_rate_record = []
    args.record_val_acc = pd.DataFrame(columns=args.dis_cvs_files)
    args.record_test_acc = pd.DataFrame(columns=args.dis_cvs_files)
    args.save_model = False # set to false donot save the intermeidate model
    args.best_eval_loss = {}
    
    for single_client in args.dis_cvs_files:
        if mode == 'pretrain':
            args.best_mlm_acc[single_client] = 0 
            args.current_mlm_acc[single_client] = []

        if mode == 'finetune':
            args.best_acc[single_client] = 0 if args.nb_classes > 1 else 999
            args.current_acc[single_client] = 0
            args.current_test_acc[single_client] = []
            args.best_eval_loss[single_client] = 9999


def crop_top(img, percent=0.15):
    offset = int(img.shape[0] * percent)
    return img[offset:]

def central_crop(img):
    size = min(img.shape[0], img.shape[1])
    offset_h = int((img.shape[0] - size) / 2)
    offset_w = int((img.shape[1] - size) / 2)
    return img[offset_w:offset_w + size, offset_h:offset_h + size]

def process_covidx_image(img, size=224, top_percent=0.08, crop=False):
    img = crop_top(img, percent=top_percent)
    if crop:
        img = central_crop(img)
    img = resize(img, (size, size))
    img = img * 255
    return img

def process_covidx_image_v2(img, size=224):
    img = cv2.resize(img, (size, size))
    img = img.astype('float64')
    img -= img.mean()
    img /= img.std()
    return img
    
def random_ratio_resize(img, prob=0.3, delta=0.1):
    if np.random.rand() >= prob:
        return img
    ratio = img.shape[0] / img.shape[1]
    ratio = np.random.uniform(max(ratio - delta, 0.01), ratio + delta)

    if ratio * img.shape[1] <= img.shape[1]:
        size = (int(img.shape[1] * ratio), img.shape[1])
    else:
        size = (img.shape[0], int(img.shape[0] / ratio))

    dh = img.shape[0] - size[1]
    top, bot = dh // 2, dh - dh // 2
    dw = img.shape[1] - size[0]
    left, right = dw // 2, dw - dw // 2

    if size[0] > 224 or size[1] > 224:
        print(img.shape, size, ratio)
    
    img = cv2.resize(img, size)
    
    padding = (left, top, right, bot)
    new_im = ImageOps.expand(img, padding)
    
    return img
