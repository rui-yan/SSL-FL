# --------------------------------------------------------
# Based on BEiT code bases
# Integrate BEiT for Federated Learning
# Reference: https://github.com/microsoft/unilm/tree/master/beit
# Author: Rui Yan
# --------------------------------------------------------

import math
import sys
from typing import Iterable

import torch
import torch.nn as nn

import sys
sys.path.insert(0,'..')
import util.misc as misc

def train_one_epoch(args, model: torch.nn.Module, d_vae: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, 
                    cur_single_client,
                    max_norm: float = 0, 
                    proxy_single_client=None,
                    log_writer=None,
                    criterion=None,
                    lr_scheduler=None, start_steps=None,
                    lr_schedule_values=None, wd_schedule_values=None):
    model.train()
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10
    
    metric_logger.log_every(data_loader, print_freq, header)
    
    for step, (batch, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # assign learning rate & weight decay for each step
        args.global_step_per_client[proxy_single_client] += 1
        if lr_schedule_values is not None or wd_schedule_values is not None:
            for it, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]

        samples, images, bool_masked_pos = batch
        images = images.to(device, non_blocking=True)
        samples = samples.to(device, non_blocking=True)
        bool_masked_pos = bool_masked_pos.to(device, non_blocking=True)

        with torch.no_grad():
            input_ids = d_vae.get_codebook_indices(images).flatten(1)
            bool_masked_pos = bool_masked_pos.flatten(1).to(torch.bool)
            labels = input_ids[bool_masked_pos]

        with torch.cuda.amp.autocast():
            outputs = model(samples, bool_masked_pos=bool_masked_pos, return_all_tokens=False)
            loss = criterion(input=outputs, target=labels)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()
        
        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
                                parameters=model.parameters(),
                                create_graph=is_second_order)
        loss_scale_value = loss_scaler.state_dict()["scale"]

        torch.cuda.synchronize()
        
        mlm_acc = (outputs.max(-1)[1] == labels).float().mean().item()
        
        metric_logger.update(mlm_acc=mlm_acc)
        metric_logger.update(loss=loss_value)
        metric_logger.update(loss_scale=loss_scale_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)
        weight_decay_value = None
        for group in optimizer.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]
        metric_logger.update(weight_decay=weight_decay_value)
        metric_logger.update(grad_norm=grad_norm)
        
        if lr_scheduler is not None:
            lr_scheduler.step_update(start_steps + step)
    
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    
    if log_writer is not None:
        for k, v in metric_logger.meters.items():
            if k in ['lr', 'min_lr', 'weight_decay', 'grad_norm']:
                log_writer.writer.add_scalar(proxy_single_client +'/opt/'+ k, v.global_avg, log_writer.step)  
            elif k in ['loss', 'mlm_acc', 'loss_scale']:
                log_writer.writer.add_scalar(proxy_single_client +'/loss/'+ k, v.global_avg, log_writer.step) 
            
            log_writer.set_step()
    
    args.current_mlm_acc[cur_single_client] = metric_logger.get_mlm_acc()
    if args.best_mlm_acc[cur_single_client] < args.current_mlm_acc[cur_single_client]:
        args.best_mlm_acc[cur_single_client] = args.current_mlm_acc[cur_single_client]
    
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
