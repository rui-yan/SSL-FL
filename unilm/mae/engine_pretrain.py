# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import math
import sys
from typing import Iterable

import torch

import util.misc as misc
import util.lr_sched as lr_sched


def train_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    cur_single_client,
                    proxy_single_client=None,
                    log_writer=None,
                    args=None):
    
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20
    
    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (samples, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        
        args.global_step_per_client[proxy_single_client] += 1
        
        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples = samples.to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
            loss, _, _ = model(samples, mask_ratio=args.mask_ratio)

        loss_value = loss.item()
        
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        # if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
        #     """ We use epoch_1000x as the x-axis in tensorboard.
        #     This calibrates different curves when batch size changes.
        #     """
        #     epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
        #     log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
        #     log_writer.add_scalar('lr', lr, epoch_1000x)
    
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    
    if log_writer is not None:
        for k, v in metric_logger.meters.items():
            if k in ['lr']:
                log_writer.writer.add_scalar(proxy_single_client +'/opt/'+ k, v.global_avg, log_writer.step)  
            elif k in ['loss']:
                log_writer.writer.add_scalar(proxy_single_client +'/loss/'+ k, v.global_avg, log_writer.step) 
            
#             log_writer.update(loss=loss_value, head=proxy_single_client + "/loss")
#             log_writer.update(loss_scale=loss_scale_value, head=proxy_single_client + "/opt")
#             log_writer.update(lr=max_lr, head=proxy_single_client + "/opt")
#             log_writer.update(min_lr=min_lr, head=proxy_single_client + "/opt")
#             log_writer.update(weight_decay=weight_decay_value, head=proxy_single_client + "/opt")
#             log_writer.update(grad_norm=grad_norm, head=proxy_single_client + "/opt")
            
            log_writer.set_step()
#             print('step: ', self.step)
#             print('global_step_per_client: ', args.global_step_per_client[proxy_single_client])
    
    # args.current_mlm_acc[cur_single_client] = metric_logger.get_mlm_acc()
    # if args.best_mlm_acc[cur_single_client] < args.current_mlm_acc[cur_single_client]:
    #     args.best_mlm_acc[cur_single_client] = args.current_mlm_acc[cur_single_client]
    
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}