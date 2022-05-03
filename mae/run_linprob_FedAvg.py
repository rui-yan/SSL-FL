# --------------------------------------------------------
# Based on MAE code bases
# Integrate MAE for Federated Learning
# Reference: https://github.com/facebookresearch/mae
# Author: Rui Yan
# --------------------------------------------------------

import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

import timm

assert timm.__version__ == "0.3.2" # version check
from timm.models.layers import trunc_normal_
from timm.data.mixup import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy

import models_vit

from engine_finetune import train_one_epoch, evaluate
from copy import deepcopy

import sys
sys.path.insert(1, '/home/yan/SSL-FL/')

import util.misc as misc
from FedAvg_utils.util import Partial_Client_Selection, valid, average_model
from FedAvg_utils.data_utils import DatasetFLFinetune, create_dataset_and_evalmetrix
from FedAvg_utils.start_config import print_options

def get_args():
    parser = argparse.ArgumentParser('MAE fine-tuning for image classification', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--save_ckpt_freq', default=20, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model', default='vit_large_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')
    
    parser.add_argument('--model_name', default='mae', type=str)
    
    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--layer_decay', type=float, default=0.75,
                        help='layer-wise lr decay from ELECTRA/BEiT')

    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR')

    # * Finetuning params
    parser.add_argument('--finetune', default='',
                        help='finetune from checkpoint')
    parser.add_argument('--global_pool', action='store_true')
    parser.set_defaults(global_pool=True)
    parser.add_argument('--cls_token', action='store_false', dest='global_pool',
                        help='Use class token instead of global pool for classification')

    # Dataset parameters\
    parser.add_argument('--data_set', default='IMNET', choices=['CIFAR10', 'COVIDfl', 'ISIC', 
                                                                'IMNET', 'Retina', 'image_folder'],
                        type=str, help='dataset for pretraining')
    parser.add_argument('--data_path', default='/datasets01/imagenet_full_size/061417/', type=str,
                        help='dataset path')
    parser.add_argument('--nb_classes', default=1000, type=int,
                        help='number of the classification types')

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default=None,
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')
    
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true',
                        help='Perform evaluation only')
    parser.add_argument('--dist_eval', action='store_true', default=False,
                        help='Enabling distributed evaluation (recommended during training for faster monitor')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)
    
    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--sync_bn', default=False, action='store_true')
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    
    # FL related parameters
    parser.add_argument("--n_clients", default=5, type=int, help="Number of clients")
    parser.add_argument("--E_epoch", default=1, type=int, help="Local training epoch in FL")
    parser.add_argument("--max_communication_rounds", default=100, type=int,
                        help="Total communication rounds.")
    parser.add_argument("--num_local_clients", default=10, choices=[10, -1], type=int, 
                        help="Num of local clients joined in each FL train. -1 indicates all clients")
    parser.add_argument("--split_type", type=str, choices=["split_1", "split_2", "split_3", "split_real", "central"], 
                        default="central", help="Which data partitions to use")

    return parser.parse_args()


def main(args, model):
    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))
    
    device = torch.device(args.device)

    # fix the seed for reproducibility
    misc.fix_random_seeds(args)

    cudnn.benchmark = True
    
    # prepare dataset
    create_dataset_and_evalmetrix(args, mode='finetune')
    
    if args.disable_eval_during_finetuning:
        dataset_val = None
    else:
        dataset_val = DatasetFLFinetune(args=args, phase='test', mode='linprob')
    
    if args.eval:
        dataset_test = DatasetFLFinetune(args=args, phase='test', mode='linprob')
    else:
        dataset_test = None
    
    num_tasks = misc.get_world_size()
    global_rank = misc.get_rank()
    
    if args.dist_eval:
        if len(dataset_val) % num_tasks != 0:
            print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                  'This will slightly alter validation results as extra duplicate entries are added to achieve '
                  'equal num of samples per-process.')
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=True)  # shuffle=True to reduce monitor bias
    else:
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    sampler_test = torch.utils.data.SequentialSampler(dataset_test)
        
    if dataset_val is not None:
        data_loader_val = torch.utils.data.DataLoader(
            dataset_val, sampler=sampler_val,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False
        )
    else:
        data_loader_val = None
        
    if dataset_test is not None:
        data_loader_test = torch.utils.data.DataLoader(
            dataset_test, sampler=sampler_test,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False
        )
    else:
        data_loader_test = None
        
    # configuration for FedAVG, prepare model, optimizer, scheduler 
    model_all, optimizer_all, criterion_all, loss_scaler_all, mixup_fn_all = Partial_Client_Selection(args, model, mode='linprob')
    model_avg = deepcopy(model).cpu()
    
    if args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None


# ---------- Train! (use different clients)    
    print("=============== Running fine-tuning ===============")
    tot_clients = args.dis_cvs_files
    print('total_clients: ', tot_clients)
    epoch = -1
    
    start_time = time.time()
    max_accuracy = 0.0
        
    while True:
        print('epoch: ', epoch)
        epoch += 1
        
        # randomly select partial clients
        if args.num_local_clients == len(args.dis_cvs_files):
            # just use all the local clients
            cur_selected_clients = args.proxy_clients
        else:
            cur_selected_clients = np.random.choice(tot_clients, args.num_local_clients, replace=False).tolist()
        
        # Get the quantity of clients joined in the FL train for updating the clients weights
        cur_tot_client_Lens = 0
        for client in cur_selected_clients:
            cur_tot_client_Lens += args.clients_with_len[client]

        for cur_single_client, proxy_single_client in zip(cur_selected_clients, args.proxy_clients):
            print('cur_single_client: ', cur_single_client)
            print('proxy_single_client: ', proxy_single_client)
            
            args.single_client = cur_single_client
            args.clients_weightes[proxy_single_client] = args.clients_with_len[cur_single_client] / cur_tot_client_Lens
            
            # ---- get dataset for each client for pretraining finetuning 
            dataset_train = DatasetFLFinetune(args=args, phase='train', mode='linprob')
            
            num_tasks = misc.get_world_size()
            global_rank = misc.get_rank()
            
            print(f'=========client: {proxy_single_client} ==============')
            if args.distributed:
                sampler_train = torch.utils.data.DistributedSampler(
                    dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
                )
            else:    
                sampler_train = torch.utils.data.RandomSampler(dataset_train)
                    
            print("Sampler_train = %s" % str(sampler_train))
            
            data_loader_train = torch.utils.data.DataLoader(
                dataset_train, sampler=sampler_train,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                pin_memory=args.pin_mem,
                drop_last=True,
            )
            
            # ---- prepare model for a client
            model = model_all[proxy_single_client]
            optimizer = optimizer_all[proxy_single_client]
            criterion = criterion_all[proxy_single_client]
            loss_scaler = loss_scaler_all[proxy_single_client]
            mixup_fn = mixup_fn_all[proxy_single_client]
            
            if args.distributed:
                model_without_ddp = model.module
            else:
                model_without_ddp = model
            
            model_ema = None
            # if args.model_ema:
            #     # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
            #     model_ema = ModelEma(
            #         model,
            #         decay=args.model_ema_decay,
            #         device='cpu' if args.model_ema_force_cpu else '',
            #         resume='')
            #     print("Using EMA with decay = %.8f" % args.model_ema_decay)
            
            n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
            # print("Model = %s" % str(model_without_ddp))
            # print('number of params:', n_parameters)
            total_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
            num_training_steps_per_inner_epoch = len(dataset_train) // total_batch_size
            print("LR = %.8f" % args.lr)
            print("Batch size = %d" % total_batch_size)
            print("Number of training examples = %d" % len(dataset_train))
            print("Number of training training per epoch = %d" % num_training_steps_per_inner_epoch)
            
            if args.distributed:
                data_loader_train.sampler.set_epoch(epoch)
            if log_writer is not None:
                log_writer.set_step(epoch)
            
            if args.eval:
                misc.load_model(args=args, model_without_ddp=model_without_ddp,
                                optimizer=optimizer, loss_scaler=loss_scaler, model_ema=model_ema)
                
                test_stats = valid(args, model, data_loader_test)
                print(f"Accuracy of the network on the {len(dataset_test)} test images: {test_stats['acc1']:.1f}%")
                model.cpu()
                
                exit(0)
            
            for inner_epoch in range(args.E_epoch):
                # ============ training one epoch of BEiT  ============
                train_stats = train_one_epoch(
                        model, criterion, data_loader_train,
                        optimizer, device, epoch, loss_scaler,
                        proxy_single_client,
                        log_writer=log_writer,
                        args=args
                        )
                
                # ============ writing logs ============
                log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                             'client': cur_single_client,
                             'epoch': epoch,
                             'inner_epoch': inner_epoch,
                             'n_parameters': n_parameters}
                
                if args.output_dir and misc.is_main_process():
                    if log_writer is not None:
                        log_writer.flush()
                    with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                        f.write(json.dumps(log_stats) + "\n")
            
        # =========== model average and eval ============ 
        # average model
        average_model(args, model_avg, model_all)
        
        # save the global model
        # TO CHECK: global model is the same for each client?
        if args.output_dir:
            if (epoch + 1) % args.save_ckpt_freq == 0 or epoch + 1 == args.max_communication_rounds:
                misc.save_model(
                    args=args, model=model_avg, model_without_ddp=model_avg,
                    optimizer=optimizer, loss_scaler=loss_scaler, epoch=epoch)
        
        if data_loader_val is not None:
            model_avg.to(args.device)
            test_stats = valid(args, model_avg, data_loader_val)
            print(f"Accuracy of the network on the {len(dataset_val)} validation images: {test_stats['acc1']:.1f}%")
            
            if max_accuracy < test_stats["acc1"]:
                max_accuracy = test_stats["acc1"]
                if args.output_dir:
                    misc.save_model(
                        args=args, model=model_avg, 
                        model_without_ddp=model_without_ddp, optimizer=optimizer,
                        loss_scaler=loss_scaler, epoch="best", model_ema=model_ema)
                
            print(f'Max accuracy: {max_accuracy:.2f}%')
            if log_writer is not None:
                log_writer.update(test_acc1=test_stats['acc1'], head="perf", step=epoch)
                log_writer.update(test_acc5=test_stats['acc5'], head="perf", step=epoch)
                log_writer.update(test_loss=test_stats['loss'], head="perf", step=epoch)
            
            log_stats = {**{f'test_{k}': v for k, v in test_stats.items()},
                         'epoch': epoch,
                         'n_parameters': n_parameters}
            
        if args.output_dir and misc.is_main_process():
                if log_writer is not None:
                    log_writer.flush()
                with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                    f.write(json.dumps(log_stats) + "\n")
        
        model_avg.to('cpu')
        
        print('global_step_per_client: ', args.global_step_per_client[proxy_single_client])
        print('t_total: ', args.t_total[proxy_single_client])
        
        if args.global_step_per_client[proxy_single_client] >= args.t_total[proxy_single_client]:
            total_time = time.time() - start_time
            total_time_str = str(datetime.timedelta(seconds=int(total_time)))
            print('Training time {}'.format(total_time_str))
            break


if __name__ == '__main__':
    args = get_args()

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    model = models_vit.__dict__[args.model](
        num_classes=args.nb_classes,
        global_pool=args.global_pool,
        )

    print_options(args, model)
    
    # set train val related paramteres
    args.best_acc = {}
    args.current_acc = {}
    args.current_test_acc = {}
    
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        
    # run finetuning
    main(args, model
