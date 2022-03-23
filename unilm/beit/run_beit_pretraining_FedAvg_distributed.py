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

import modeling_pretrain
from engine_for_pretraining import train_one_epoch
from copy import deepcopy

import sys
sys.path.insert(1, '/home/yan/SSL-FL/unilm/')

import util.misc as misc

from FedAvg_utils.util import Partial_Client_Selection, valid, average_model
from FedAvg_utils.data_utils import DatasetFLPretrain, create_dataset_and_evalmetrix
from FedAvg_utils.start_config import print_options

def get_args():
    parser = argparse.ArgumentParser('BEiT pre-training', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--save_ckpt_freq', default=20, type=int)
    parser.add_argument("--discrete_vae_weight_path", type=str)
    parser.add_argument("--discrete_vae_type", type=str, default="dall-e")
    
    # Model parameters
    parser.add_argument('--model_name', default='beit', type=str)
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
    parser.add_argument('--data_set', default='IMNET', choices=['CIFAR10', 'COVIDfl', 'ISIC', 
                                                                'IMNET', 'Retina', 'image_folder'],
                        type=str, help='dataset for pretraining')
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
    parser.add_argument('--sync_bn', default=False, action='store_true')
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    
    # FL related parameters
    parser.add_argument("--n_clients", default=5, type=int, help="Number of clients")
    parser.add_argument("--E_epoch", default=1, type=int, help="Local training epoch in FL")
    parser.add_argument("--max_communication_rounds", default=100, type=int,
                        help="Total communication rounds")
    parser.add_argument("--num_local_clients", default=10, choices=[10, -1], type=int, 
                        help="Num of local clients joined in each FL train. -1 indicates all clients")
    parser.add_argument("--split_type", type=str, choices=["split_1", "split_2", "split_3", "split_real", "central"], 
                        default="central", help="Which data partitions to use")
    
    return parser.parse_args()


def get_model(args):
    print(f"Creating model: {args.model}")
    model = create_model(
        args.model,
        pretrained=False,
        drop_path_rate=args.drop_path,
        drop_block_rate=None,
        use_shared_rel_pos_bias=args.rel_pos_bias,
        use_abs_pos_emb=args.abs_pos_emb,
        init_values=args.layer_scale_init_value,
    )
    
    # set patch_size and window_size for beit pretraining
    patch_size = model.patch_embed.patch_size
    print("patch size = %s" % str(patch_size))
    args.window_size = (args.input_size // patch_size[0], args.input_size // patch_size[1])
    args.patch_size = patch_size
    
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print("Model = %s" % str(model))
    # print('number of params:', n_parameters)
    
    return model


def main(args, model):
    
    misc.init_distributed_mode(args)
    
    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))
    
    device = torch.device(args.device)
    
    # fix the seed for reproducibility
    misc.fix_random_seeds(args)
    
    cudnn.benchmark = True
    
    # prepare output_dir to save model checkpoints
    os.makedirs(args.output_dir, exist_ok=True)
    
    # prepare dataset
    create_dataset_and_evalmetrix(args) 
    
    # configuration for FedAVG, prepare model, optimizer, scheduler 
    model_all, optimizer_all, criterion_all, lr_scheduler_all, wd_scheduler_all, loss_scaler_all = Partial_Client_Selection(args, model)
    model_avg = deepcopy(model).cpu()
    
    # prepare discrete vae
    d_vae = misc.create_d_vae(
        weight_path=args.discrete_vae_weight_path, d_vae_type=args.discrete_vae_type,
        device=device, image_size=args.second_input_size)
    
    global_rank = misc.get_rank()
    
    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = misc.TensorboardLogger(log_dir=args.log_dir)
    else:
        log_writer = None
    
    # ---------- Train! (use different clients)
    print("=============== Running pre-training ===============")
    tot_clients = args.dis_cvs_files
    print('total_clients: ', tot_clients)
    epoch = -1
    
    print(f"Start training for {args.max_communication_rounds} epochs, distributed={args.distributed}")
    start_time = time.time()
    
    while True:
        print('epoch: ', epoch)
        epoch += 1
        
        # randomly select partial clients
        if args.num_local_clients == len(args.dis_cvs_files):
            # just use all the local clients
            cur_selected_clients = args.proxy_clients
        else:
            cur_selected_clients = np.random.choice(tot_clients, args.num_local_clients, replace=False).tolist()
        
        # get the quantity of clients joined in the FL train for updating the clients weights
        cur_tot_client_Lens = 0
        for client in cur_selected_clients:
            cur_tot_client_Lens += args.clients_with_len[client]
        
        for cur_single_client, proxy_single_client in zip(cur_selected_clients, args.proxy_clients):
            print('cur_single_client: ', cur_single_client)
            print('proxy_single_client: ', proxy_single_client)
            
            args.single_client = cur_single_client
            args.clients_weightes[proxy_single_client] = args.clients_with_len[cur_single_client] / cur_tot_client_Lens
            
            # ---- get dataset for each client for pretraining
            dataset_train = DatasetFLPretrain(args)
            
            num_tasks = misc.get_world_size()
            global_rank = misc.get_rank()
            sampler_rank = global_rank
            num_training_steps_per_inner_epoch = len(dataset_train) // args.batch_size // num_tasks
            
            print(f'=========client: {proxy_single_client} ==============')
            if args.distributed:
                sampler_train = torch.utils.data.DistributedSampler(
                     dataset_train, num_replicas=num_tasks, rank=sampler_rank, shuffle=True)
            else:    
                sampler_train = torch.utils.data.RandomSampler(dataset_train)

            data_loader_train = torch.utils.data.DataLoader(
                dataset_train, sampler=sampler_train,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                pin_memory=args.pin_mem,
                drop_last=True,
                )
            # print('train_data_len; ', len(data_loader_train))
            
            # ---- prepare model for a client
            model = model_all[proxy_single_client]
            optimizer = optimizer_all[proxy_single_client]
            criterion = criterion_all[proxy_single_client]
            lr_schedule_values = lr_scheduler_all[proxy_single_client]
            wd_schedule_values = wd_scheduler_all[proxy_single_client]
            loss_scaler = loss_scaler_all[proxy_single_client]
            
            # print('model: ', model.module.named_parameters())
            # print('device :', next(model.module.parameters()).device)
            # model = model.to(device)
            
            if args.distributed:
                model_without_ddp = model.module
            else:
                model_without_ddp = model
            
            # print("Model = %s" % str(model))
            # print("Model_without_ddp = %s" % str(model_without_ddp))
            
            if args.distributed:
                data_loader_train.sampler.set_epoch(epoch)
            if log_writer is not None:
                log_writer.set_step(epoch)
            
            n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)            
            total_batch_size = args.batch_size * num_tasks
            print("LR = %.8f" % args.lr)
            print("Batch size = %d" % total_batch_size)
            print("Number of training steps = %d" % num_training_steps_per_inner_epoch)
            print("Number of training examples per epoch = %d" % (total_batch_size * num_training_steps_per_inner_epoch))
            
            for inner_epoch in range(args.E_epoch):
                # ============ training one epoch of BEiT  ============
                train_stats = train_one_epoch(args, model, d_vae, data_loader_train,
                                              optimizer, device, epoch, 
                                              loss_scaler=loss_scaler,
                                              cur_single_client=cur_single_client,
                                              max_norm=args.clip_grad, 
                                              proxy_single_client=proxy_single_client,
                                              log_writer=log_writer,
                                              criterion=criterion,
                                              start_steps=(epoch + inner_epoch) * num_training_steps_per_inner_epoch,
                                              lr_schedule_values=lr_schedule_values,
                                              wd_schedule_values=wd_schedule_values,
                                              )
                
                # ============ writing logs ============
                # if args.output_dir:
                #     if (epoch + 1) % args.save_ckpt_freq == 0 and inner_epoch + 1 == args.E_epoch:
                #         utils.save_model(
                #             args=args, model=model, model_without_ddp=model_without_ddp,
                #             optimizer=optimizer, loss_scaler=loss_scaler, 
                #             epoch=epoch)
                # TO CHECK? ONLY Save global model during average
                
                log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                             'client': args.single_client,
                             'epoch': epoch,
                             'inner_epoch': inner_epoch,
                             'n_parameters': n_parameters}
                
                if args.output_dir and misc.is_main_process():
                    if log_writer is not None:
                        log_writer.flush()
                    with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                        f.write(json.dumps(log_stats) + "\n")
            
            # we use frequent transfer of model between GPU and CPU due to limitation of GPU memory
            # model.to('cpu')
        
        # average model
        average_model(args, model_avg, model_all)
        
        # save the global model
        # TO CHECK: global model is the same for each client?
        if args.output_dir:
            if (epoch + 1) % args.save_ckpt_freq == 0:
                misc.save_model(
                    args=args, model=model_avg, model_without_ddp=model_avg,
                    optimizer=optimizer, loss_scaler=loss_scaler, epoch=epoch)
        
        # print('global_step_per_client: ', args.global_step_per_client[proxy_single_client])
        # print('t_total: ', args.t_total[proxy_single_client])
        
        # end criterion
        if args.global_step_per_client[proxy_single_client] >= args.t_total[proxy_single_client]:
            break
    
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    
    print("================End pre-training! ================ ")
    print('pretraining time {}'.format(total_time_str))


if __name__ == '__main__':
    opts = get_args()
    
    if opts.output_dir:
        Path(opts.output_dir).mkdir(parents=True, exist_ok=True)
    
    # initialize model
    model = get_model(opts)
    print_options(opts, model)
    
    # set train val related paramteres
    opts.best_mlm_acc = {}
    opts.current_mlm_acc = {}
    
    # run pretraining
    main(opts, model)
    
    # Show final performance
#     message = '\n \n ==========Start showing final performance ============ \n'
#     message += 'Final union mlm accuracy is: %2.5f with std: %2.5f \n' %  \
#                    (np.asarray(list(opts.current_mlm_acc.values())).mean(), 
#                     np.asarray(list(opts.current_mlm_acc.values())).std())
#     message += "================ End ================ \n"
    
#     with open(opts.file_name, 'a+') as args_file:
#         args_file.write(message)
#         args_file.write('\n')
    
    # print(message)
