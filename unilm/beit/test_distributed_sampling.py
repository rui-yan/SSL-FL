# --------------------------------------------------------
# BEIT: BERT Pre-Training of Image Transformers (https://arxiv.org/abs/2106.08254)
# Github source: https://github.com/microsoft/unilm/tree/master/beit
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# By Hangbo Bao
# Based on timm, DINO and DeiT code bases
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit
# https://github.com/facebookresearch/dino
# --------------------------------------------------------'
import argparse
import datetime
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
import json
import os

from pathlib import Path

from timm.models import create_model
from optim_factory import create_optimizer
from engine_for_pretraining import train_one_epoch
import utils
from FedAvg_utils.util import Partial_Client_Selection, valid, average_model
from FedAvg_utils.data_utils import DatasetFLBEiTPretrain, create_dataset_and_evalmetrix
from FedAvg_utils.start_config import print_options
import modeling_pretrain
from copy import deepcopy

from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

def get_args():
    parser = argparse.ArgumentParser('BEiT pre-training script', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--save_ckpt_freq', default=20, type=int)
    parser.add_argument("--discrete_vae_weight_path", type=str)
    parser.add_argument("--discrete_vae_type", type=str, default="dall-e")
    
    # Model parameters
    parser.add_argument('--model', default='deit_base_patch16_224', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--rel_pos_bias', action='store_true')
    parser.add_argument('--disable_rel_pos_bias', action='store_false', dest='rel_pos_bias')
    parser.set_defaults(rel_pos_bias=True)
    parser.add_argument('--abs_pos_emb', action='store_true')
    parser.set_defaults(abs_pos_emb=False)
    parser.add_argument('--layer_scale_init_value', default=0.1, type=float, 
                        help="0.1 for base, 1e-5 for large. set 0 to disable layer scale")
    
    parser.add_argument('--num_mask_patches', default=75, type=int,
                        help='number of the visual tokens/patches need be masked')
    parser.add_argument('--max_mask_patches_per_block', type=int, default=None)
    parser.add_argument('--min_mask_patches_per_block', type=int, default=16)
    
    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size for backbone')
    parser.add_argument('--second_input_size', default=112, type=int,
                        help='images input size for discrete vae')
    
    parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')
    
    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt_eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt_betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--weight_decay_end', type=float, default=None, help="""Final value of the
        weight decay. We use a cosine schedule for WD. 
        (Set the same value with args.weight_decay to keep weight decay no change)""")

    parser.add_argument('--lr', type=float, default=5e-4, metavar='LR',
                        help='learning rate (default: 5e-4)')
    parser.add_argument('--warmup_lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min_lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')

    parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--warmup_steps', type=int, default=-1, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')

    # Augmentation parameters
    parser.add_argument('--color_jitter', type=float, default=0.4, metavar='PCT',
                        help='Color jitter factor (default: 0.4)')
    parser.add_argument('--train_interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')
    parser.add_argument('--second_interpolation', type=str, default='lanczos',
                        help='Interpolation for discrete vae (random, bilinear, bicubic default: "lanczos")')

    # Dataset parameters
    parser.add_argument('--data_set', default='IMNET', choices=['CIFAR10', 'COVIDx', 'CIFAR100', 
                                                                'IMNET', 'Retina', 'image_folder'],
                        type=str, help='ImageNet dataset path')
    parser.add_argument('--data_path', default='/datasets01/imagenet_full_size/061417/', type=str,
                        help='dataset path')
    parser.add_argument('--imagenet_default_mean_and_std', default=False, action='store_true')

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default=None,
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--auto_resume', action='store_true')
    parser.add_argument('--no_auto_resume', action='store_false', dest='auto_resume')
    parser.set_defaults(auto_resume=True)

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem',
                        help='')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    
    # FL related parameters
    parser.add_argument("--n_clients", default=5, type=int, help="Number of clients")
    parser.add_argument("--E_epoch", default=1, type=int, help="Local training epoch in FL")
    parser.add_argument("--max_communication_rounds", default=100, type=int,
                        help="Total communication rounds")
    parser.add_argument("--num_local_clients", default=10, choices=[10, -1], type=int, 
                        help="Num of local clients joined in each FL train. -1 indicates all clients")
    parser.add_argument("--split_type", type=str, choices=["split_1", "split_2", "split_3", "central"], 
                        default="central", help="Which data partitions to use")
    
    return parser.parse_args()

class RandomDataset(Dataset):
    def __init__(self, size, length, local_rank, client):
        self.len = length
        print('client : ', client)
        if client == 'train_0':
            self.data = torch.stack([torch.ones(5), torch.ones(5)*2,
                                     torch.ones(5)*3,torch.ones(5)*4,
                                     torch.ones(5)*5,torch.ones(5)*6,
                                     torch.ones(5)*7,torch.ones(5)*8,
                                     torch.ones(5)*9, torch.ones(5)*10,
                                     torch.ones(5)*11,torch.ones(5)*12,
                                     torch.ones(5)*13,torch.ones(5)*14,
                                     torch.ones(5)*15,torch.ones(5)*16])
        elif client == 'train_1':
            self.data = torch.stack([torch.ones(1), torch.ones(1)*2,
                                     torch.ones(1)*3,torch.ones(1)*4,
                                     torch.ones(1)*5,torch.ones(1)*6,
                                     torch.ones(1)*7,torch.ones(1)*8,
                                     torch.ones(1)*9, torch.ones(1)*10,
                                     torch.ones(1)*11,torch.ones(1)*12,
                                     torch.ones(1)*13,torch.ones(1)*14,
                                     torch.ones(1)*15,torch.ones(1)*16])

        self.local_rank = local_rank
    def __getitem__(self, index):

        return self.data[index]

    def __len__(self):
        return self.len
    

def main(args):
        
    # fix the seed for reproducibility
    utils.init_distributed_mode(args)
    utils.fix_random_seeds(args)
    
    device = torch.device(args.device)
    
    cudnn.benchmark = True
    
    # ---------- Train! (use different clients)
    print("=============== Running pre-training ===============")
    # tot_clients = args.dis_cvs_files
    # print('total_clients: ', tot_clients)
    epoch = -1
    
    start_time = time.time()
            
    while True:
        print('eopch: ', epoch)
        epoch += 1
        
        # randomly select partial clients
        # if args.num_local_clients == len(args.dis_cvs_files):
        #     # just use all the local clients
        #     cur_selected_clients = args.proxy_clients
        # else:
        #     cur_selected_clients = np.random.choice(tot_clients, args.num_local_clients, replace=False).tolist()
        cur_selected_clients = ['train_0', 'train_1']
        args.proxy_clients = ['train_0', 'train_1']
        # get the quantity of clients joined in the FL train for updating the clients weights
        # cur_tot_client_Lens = 0
        # for client in cur_selected_clients:
        #     cur_tot_client_Lens += args.clients_with_len[client]
        
        for cur_single_client, proxy_single_client in zip(cur_selected_clients, args.proxy_clients):
            print('cur_single_client: ', cur_single_client)
            print('proxy_single_client: ', proxy_single_client)
            
            args.single_client = cur_single_client
            
            # ---- get dataset for each client for pretraining
            num_tasks = utils.get_world_size()
            global_rank = utils.get_rank()
            sampler_rank = global_rank

            dataset = RandomDataset(size=5, length=16, local_rank=sampler_rank, client=args.single_client)
            
            if args.distributed:
                sampler = torch.utils.data.DistributedSampler(
                     dataset, num_replicas=num_tasks, rank=sampler_rank, shuffle=True)
            else:    
                sampler = torch.utils.data.RandomSampler(dataset)
            
            data_loader = torch.utils.data.DataLoader(dataset, sampler=sampler, batch_size=2)
            
            if args.distributed:
                data_loader.sampler.set_epoch(epoch)

            # sampler.set_epoch(e)
            for data in data_loader:
                print(data)
        
        # end criterion
        if epoch == 2:
            break
        
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    
    print("================End pre-training! ================ ")
    print('pretraining time {}'.format(total_time_str))


if __name__ == '__main__':
    opts = get_args()
    
    # run pretraining
    main(opts)
    
