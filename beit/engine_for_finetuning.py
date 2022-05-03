# --------------------------------------------------------
# Based on BEiT code bases
# Integrate BEiT for Federated Learning
# Reference: https://github.com/microsoft/unilm/tree/master/beit
# Author: Rui Yan
# --------------------------------------------------------

import math
from typing import Iterable, Optional

import torch

from timm.data import Mixup
from timm.utils import accuracy, ModelEma

import sys
sys.path.insert(0,'..')
import util.misc as misc

def train_class_batch(model, samples, target, criterion):
    outputs = model(samples)
    loss = criterion(outputs, target)
    return loss, outputs


def get_loss_scale_for_deepspeed(model):
    optimizer = model.optimizer
    return optimizer.loss_scale if hasattr(optimizer, "loss_scale") else optimizer.cur_scale


def train_one_epoch(args, model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, 
                    cur_single_client,
                    max_norm: float = 0,
                    proxy_single_client=None,log_writer=None,
                    model_ema: Optional[ModelEma] = None, 
                    mixup_fn: Optional[Mixup] = None, 
                    start_steps=None, lr_schedule_values=None, wd_schedule_values=None,
                    num_training_steps_per_inner_epoch=None, update_freq=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    if loss_scaler is None:
        model.zero_grad()
        model.micro_steps = 0
    else:
        optimizer.zero_grad()

    for data_iter_step, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        step = data_iter_step // update_freq
        if step >= num_training_steps_per_inner_epoch:
            continue
        
        it = start_steps + step  # global training iteration
        # print('start_steps: ', start_steps)
        # print('local_step: ', step)
        # print('iter: ', it)
        args.global_step_per_client[proxy_single_client] += 1
        # args.global_step_per_client[proxy_single_client] = it
        # Update LR & WD for the first acc
        if lr_schedule_values is not None or wd_schedule_values is not None and data_iter_step % update_freq == 0:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        if loss_scaler is None:
            samples = samples.half()
            loss, output = train_class_batch(
                model, samples, targets, criterion)
        else:
            with torch.cuda.amp.autocast():
                loss, output = train_class_batch(
                    model, samples, targets, criterion)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        if loss_scaler is None:
            loss /= update_freq
            model.backward(loss)
            model.step()

            if (data_iter_step + 1) % update_freq == 0:
                # model.zero_grad()
                # Deepspeed will call step() & model.zero_grad() automatic
                if model_ema is not None:
                    model_ema.update(model)
            grad_norm = None
            loss_scale_value = get_loss_scale_for_deepspeed(model)
        else:
            # this attribute is added by timm on one optimizer (adahessian)
            is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
            loss /= update_freq
            grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
                                    parameters=model.parameters(), create_graph=is_second_order,
                                    update_grad=(data_iter_step + 1) % update_freq == 0)
            if (data_iter_step + 1) % update_freq == 0:
                optimizer.zero_grad()
                if model_ema is not None:
                    model_ema.update(model)
            loss_scale_value = loss_scaler.state_dict()["scale"]
        
        torch.cuda.synchronize()

        if mixup_fn is None:
            class_acc = (output.max(-1)[-1] == targets).float().mean()
        else:
            class_acc = None
        metric_logger.update(loss=loss_value)
        metric_logger.update(class_acc=class_acc)
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
            
        args.learning_rate_record[proxy_single_client].append(optimizer.param_groups[0]['lr'])
    
    # gather the stats from all processes
    print("Averaged stats (before sync):", metric_logger)
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    
    if log_writer is not None:
        for k, v in metric_logger.meters.items():
            if k in ['lr', 'min_lr', 'weight_decay', 'grad_norm', 'loss_scale']:
                log_writer.writer.add_scalar(proxy_single_client +'/opt/'+ k, v.global_avg, log_writer.step)  
            elif k in ['loss', 'class_acc', 'loss_scale']:
                # print('hey: ', k, v.global_avg)
                log_writer.writer.add_scalar(proxy_single_client +'/loss/'+ k, v.global_avg, log_writer.step) 
                
            # log_writer.update(loss=loss_value, head=proxy_single_client + "/loss")
            # log_writer.update(class_acc=class_acc, head=proxy_single_client + "/loss")
            # log_writer.update(loss_scale=loss_scale_value, head=proxy_single_client + "/opt")
            # log_writer.update(lr=max_lr, head=proxy_single_client + "/opt")
            # log_writer.update(min_lr=min_lr, head=proxy_single_client + "/opt")
            # log_writer.update(weight_decay=weight_decay_value, head=proxy_single_client + "/opt")
            # log_writer.update(grad_norm=grad_norm, head=proxy_single_client + "/opt")
            
            log_writer.set_step()
        
    args.current_acc[cur_single_client] = metric_logger.get_class_acc()
    
    print('best_acc:', args.best_acc[cur_single_client])
    print('current_acc:', args.current_acc[cur_single_client])
    if args.best_acc[cur_single_client] < args.current_acc[cur_single_client]:
        args.best_acc[cur_single_client] = args.current_acc[cur_single_client]
    
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device):
    criterion = torch.nn.CrossEntropyLoss()
    
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    for batch in metric_logger.log_every(data_loader, 10, header):
        images = batch[0]
        target = batch[-1]
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            output = model(images)
            loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    # metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
