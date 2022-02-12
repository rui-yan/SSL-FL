from __future__ import absolute_import, division, print_function
import os
import numpy as np
from copy import deepcopy

import torch
import torch.nn as nn
import sys
sys.path.insert(0,'..')

import utils
from utils import NativeScalerWithGradNormCount as NativeScaler
from timm.utils import accuracy, ModelEma
from timm.data.mixup import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from optim_factory import create_optimizer, get_parameter_groups, LayerDecayValueAssigner

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def save_model(args, model):
    model_to_save = model.module if hasattr(model, 'module') else model
    client_name = os.path.basename(args.single_client).split('.')[0]
    model_checkpoint = os.path.join(args.output_dir, "%s_%s_checkpoint.bin" % (args.name, client_name))
    
    torch.save(model_to_save.state_dict(), model_checkpoint)
    # print("Saved model checkpoint to [DIR: %s]", args.output_dir)


def inner_valid(args, model, data_loader):
    # eval_losses = AverageMeter()
    criterion = torch.nn.CrossEntropyLoss()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    # switch to evaluation mode
    model.eval()
    
    print("++++++ Running Validation ++++++")
    for batch in metric_logger.log_every(data_loader, 10, header):
        images = batch[0]
        target = batch[-1]
        images = images.to(args.device, non_blocking=True)
        target = target.to(args.device, non_blocking=True)

        # compute output
        # with torch.cuda.amp.autocast():
        with torch.no_grad():
            output = model(images)
            loss = criterion(output, target)

        acc1, _ = accuracy(output, target, topk=(1, 2))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        # metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    # metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1,losses=metric_logger.loss))
    # print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          # .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))
    
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

#     all_preds, all_label = [], []
    
#     loss_fct = torch.nn.CrossEntropyLoss()
#     for batch in test_loader:
#         batch = tuple(t.to(args.device) for t in batch)
#         x, y = batch

#         with torch.no_grad():
#             # if not args.Use_ResNet:
#             #     logits = model(x)[0]
#             # else:
#             logits = model(x)
            
#             if args.nb_classes > 1:
#                 eval_loss = loss_fct(logits, y)
#                 eval_losses.update(eval_loss.item())

#             if args.nb_classes > 1:
#                 preds = torch.argmax(logits, dim=-1)
#             else:
#                 preds = logits

#         if len(all_preds) == 0:
#             all_preds.append(preds.detach().cpu().numpy())
#             all_label.append(y.detach().cpu().numpy())
#         else:
#             all_preds[0] = np.append(
#                 all_preds[0], preds.detach().cpu().numpy(), axis=0
#             )
#             all_label[0] = np.append(
#                 all_label[0], y.detach().cpu().numpy(), axis=0
#             )
#     all_preds, all_label = all_preds[0], all_label[0]
#     if not args.nb_classes == 1:
#         eval_result = simple_accuracy(all_preds, all_label)
#     else:
#         eval_result =  mean_squared_error(all_preds, all_label)

    # model.train()

    # return eval_result, eval_losses


def metric_evaluation(args, eval_result):
    if args.nb_classes == 1:
        if args.best_acc[args.single_client] < eval_result:
            Flag = False
        else:
            Flag = True
    else:
        if args.best_acc[args.single_client] < eval_result:
            Flag = True
        else:
            Flag = False
    return Flag


def valid(args, model, data_loader_val, data_loader_test = None, TestFlag = False):
    # Validation!
    return inner_valid(args, model, data_loader_val)
    
    # eval_result, eval_losses = inner_valid(args, model, data_loader_val)
    
#     print("Valid Loss: %2.5f" % eval_losses.avg, "Valid Accuracy: %2.5f" % eval_result)
#     if metric_evaluation(args, eval_result):
#         # if args.save_model_flag:
#         #     save_model(args, model)
        
#         args.best_acc[args.single_client] = eval_result
#         args.best_eval_loss[args.single_client] = eval_losses.val
#         print("The updated best acc of client", args.single_client, args.best_acc[args.single_client])

#         if TestFlag:
#             test_result, eval_losses = inner_valid(args, model, data_loader_test)
#             args.current_test_acc[args.single_client] = test_result
#             print('We also update the test acc of client', args.single_client, 'as',
#                   args.current_test_acc[args.single_client])
#     else:
#         print("Donot replace previous best acc of client", args.best_acc[args.single_client])

#     args.current_acc[args.single_client] = eval_result


def Partial_Client_Selection(args, model, mode='pretrain'):

    device = torch.device(args.device)
        
    # Select partial clients join in FL train
    if args.num_local_clients == -1: # all the clients joined in the train
        args.proxy_clients = args.dis_cvs_files
        args.num_local_clients =  len(args.dis_cvs_files)# update the true number of clients
    else:
        args.proxy_clients = ['train_' + str(i) for i in range(args.num_local_clients)]
    
    # Generate model for each client
    model_all = {}
    optimizer_all = {}
    criterion_all = {}
    lr_scheduler_all = {}
    wd_scheduler_all = {}
    loss_scaler_all = {}
    mixup_fn_all = {}
    args.learning_rate_record = {}
    args.t_total = {}
    
    # Load pretrained model if mode='finetune'
    if mode=='finetune' and args.finetune:
        if args.finetune.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.finetune, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.finetune, map_location='cpu')
        
        print("Load ckpt from %s" % args.finetune)
        checkpoint_model = None
        for model_key in args.model_key.split('|'):
            if model_key in checkpoint:
                checkpoint_model = checkpoint[model_key]
                print("Load state_dict by model_key = %s" % model_key)
                break
        if checkpoint_model is None:
            checkpoint_model = checkpoint
        state_dict = model.state_dict()
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        if model.use_rel_pos_bias and "rel_pos_bias.relative_position_bias_table" in checkpoint_model:
            print("Expand the shared relative position embedding to each transformer block. ")
            num_layers = model.get_num_layers()
            rel_pos_bias = checkpoint_model["rel_pos_bias.relative_position_bias_table"]
            for i in range(num_layers):
                checkpoint_model["blocks.%d.attn.relative_position_bias_table" % i] = rel_pos_bias.clone()

            checkpoint_model.pop("rel_pos_bias.relative_position_bias_table")

        all_keys = list(checkpoint_model.keys())
        for key in all_keys:
            if "relative_position_index" in key:
                checkpoint_model.pop(key)

            if "relative_position_bias_table" in key:
                rel_pos_bias = checkpoint_model[key]
                src_num_pos, num_attn_heads = rel_pos_bias.size()
                dst_num_pos, _ = model.state_dict()[key].size()
                dst_patch_shape = model.patch_embed.patch_shape
                if dst_patch_shape[0] != dst_patch_shape[1]:
                    raise NotImplementedError()
                num_extra_tokens = dst_num_pos - (dst_patch_shape[0] * 2 - 1) * (dst_patch_shape[1] * 2 - 1)
                src_size = int((src_num_pos - num_extra_tokens) ** 0.5)
                dst_size = int((dst_num_pos - num_extra_tokens) ** 0.5)
                if src_size != dst_size:
                    print("Position interpolate for %s from %dx%d to %dx%d" % (
                        key, src_size, src_size, dst_size, dst_size))
                    extra_tokens = rel_pos_bias[-num_extra_tokens:, :]
                    rel_pos_bias = rel_pos_bias[:-num_extra_tokens, :]

                    def geometric_progression(a, r, n):
                        return a * (1.0 - r ** n) / (1.0 - r)

                    left, right = 1.01, 1.5
                    while right - left > 1e-6:
                        q = (left + right) / 2.0
                        gp = geometric_progression(1, q, src_size // 2)
                        if gp > dst_size // 2:
                            right = q
                        else:
                            left = q

                    # if q > 1.090307:
                    #     q = 1.090307

                    dis = []
                    cur = 1
                    for i in range(src_size // 2):
                        dis.append(cur)
                        cur += q ** (i + 1)

                    r_ids = [-_ for _ in reversed(dis)]

                    x = r_ids + [0] + dis
                    y = r_ids + [0] + dis

                    t = dst_size // 2.0
                    dx = np.arange(-t, t + 0.1, 1.0)
                    dy = np.arange(-t, t + 0.1, 1.0)

                    print("Original positions = %s" % str(x))
                    print("Target positions = %s" % str(dx))

                    all_rel_pos_bias = []

                    for i in range(num_attn_heads):
                        z = rel_pos_bias[:, i].view(src_size, src_size).float().numpy()
                        f = interpolate.interp2d(x, y, z, kind='cubic')
                        all_rel_pos_bias.append(
                            torch.Tensor(f(dx, dy)).contiguous().view(-1, 1).to(rel_pos_bias.device))

                    rel_pos_bias = torch.cat(all_rel_pos_bias, dim=-1)

                    new_rel_pos_bias = torch.cat((rel_pos_bias, extra_tokens), dim=0)
                    checkpoint_model[key] = new_rel_pos_bias

        # interpolate position embedding
        if 'pos_embed' in checkpoint_model:
            pos_embed_checkpoint = checkpoint_model['pos_embed']
            embedding_size = pos_embed_checkpoint.shape[-1]
            num_patches = model.patch_embed.num_patches
            num_extra_tokens = model.pos_embed.shape[-2] - num_patches
            # height (== width) for the checkpoint position embedding
            orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
            # height (== width) for the new position embedding
            new_size = int(num_patches ** 0.5)
            # class_token and dist_token are kept unchanged
            if orig_size != new_size:
                print("Position interpolate from %dx%d to %dx%d" % (orig_size, orig_size, new_size, new_size))
                extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
                # only the position tokens are interpolated
                pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
                pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
                pos_tokens = torch.nn.functional.interpolate(
                    pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
                pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
                new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
                checkpoint_model['pos_embed'] = new_pos_embed

        utils.load_state_dict(model, checkpoint_model, prefix=args.model_prefix)
        
    if args.distributed:
        if args.sync_bn:
            "activate synchronized batch norm"
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    
    for proxy_single_client in args.proxy_clients:
        # model_all
        model_all[proxy_single_client] = deepcopy(model)
        model_all[proxy_single_client] = model_all[proxy_single_client].to(device)
        model_all[proxy_single_client] = torch.nn.parallel.DistributedDataParallel(model_all[proxy_single_client], device_ids=[args.gpu], find_unused_parameters=True)
        
        if args.distributed:
            model_without_ddp = model_all[proxy_single_client].module
        else:
            model_without_ddp = model_all[proxy_single_client]
        
        # optimizer_all
        if mode == 'pretrain':
            optimizer_all[proxy_single_client] = create_optimizer(args, model_without_ddp)
        
        else: # mode == 'finetune'
            num_layers = model_without_ddp.get_num_layers()
            
            if args.layer_decay < 1.0:
                assigner = LayerDecayValueAssigner(list(args.layer_decay ** (num_layers + 1 - i) for i in range(num_layers + 2)))
            else:
                assigner = None

            if assigner is not None:
                print("Assigned values = %s" % str(assigner.values))
            
            skip_weight_decay_list = model_without_ddp.no_weight_decay()
            if args.disable_weight_decay_on_rel_pos_bias:
                for i in range(num_layers):
                    skip_weight_decay_list.add("blocks.%d.attn.relative_position_bias_table" % i)
            
            optimizer_all[proxy_single_client] = create_optimizer(args, model_without_ddp,
                                                          skip_list=skip_weight_decay_list,
                                                          get_num_layer=assigner.get_layer_id if assigner is not None else None, 
                                                          get_layer_scale=assigner.get_scale if assigner is not None else None)
        
        global_rank = utils.get_rank()
        num_tasks = utils.get_world_size()
        if mode == 'pretrain':
            print('clients_with_len: ', args.clients_with_len[proxy_single_client])
            num_training_steps_per_inner_epoch = args.clients_with_len[proxy_single_client] // args.batch_size // num_tasks

            total_batch_size = args.batch_size * num_tasks
            print("Batch size = %d" % total_batch_size)
            print("Number of training steps = %d" % num_training_steps_per_inner_epoch)
            print("Number of training examples per epoch = %d" % (total_batch_size * num_training_steps_per_inner_epoch))
        else:
            print('clients_with_len: ', args.clients_with_len)
            print('clients_with_len: ', args.clients_with_len[proxy_single_client])
            total_batch_size = args.batch_size * args.update_freq * num_tasks
            num_training_steps_per_inner_epoch = args.clients_with_len[proxy_single_client] // total_batch_size

            print("Batch size = %d" % total_batch_size)
            print("Number of training steps = %d" % num_training_steps_per_inner_epoch)
            print("Number of training examples per epoch = %d" % (total_batch_size * num_training_steps_per_inner_epoch))

        # get the total decay steps first
        args.t_total[proxy_single_client] = num_training_steps_per_inner_epoch * args.E_epoch * args.max_communication_rounds
        
        # criterion_all
        if mode == 'pretrain':
            criterion_all[proxy_single_client] = nn.CrossEntropyLoss()
        else:
            mixup_fn = None
            mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
            if mixup_active:
                print("Mixup is activated!")
                mixup_fn = Mixup(
                    mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
                    prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
                    label_smoothing=args.smoothing, num_classes=args.nb_classes)
            mixup_fn_all[proxy_single_client] = mixup_fn
            
            if mixup_fn is not None:
                # smoothing is handled with mixup label transform
                criterion = SoftTargetCrossEntropy()
            elif args.smoothing > 0.:
                criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
            else:
                criterion = torch.nn.CrossEntropyLoss()
            criterion_all[proxy_single_client] = criterion
            
        # lr_scheduler_all
        print("Use step level LR & WD scheduler!")
        lr_scheduler_all[proxy_single_client] = utils.cosine_scheduler(args.lr, args.min_lr, 
                                                       epochs=args.E_epoch, 
                                                       niter_per_ep=num_training_steps_per_inner_epoch,
                                                       max_communication_rounds=args.max_communication_rounds,
                                                       warmup_epochs=args.warmup_epochs, 
                                                       warmup_steps=args.warmup_steps,)
        
        # wd_schedule_all
        if args.weight_decay_end is None:
            args.weight_decay_end = args.weight_decay
        wd_scheduler_all[proxy_single_client] = utils.cosine_scheduler(args.weight_decay, 
                                                                       args.weight_decay_end,
                                                       epochs=args.E_epoch, 
                                                       niter_per_ep=num_training_steps_per_inner_epoch,
                                                       max_communication_rounds=args.max_communication_rounds)
        
        # loss_scaler_all
        loss_scaler_all[proxy_single_client] = NativeScaler()
        
        args.learning_rate_record[proxy_single_client] = []
    
    args.clients_weightes = {}
    args.global_step_per_client = {name: 0 for name in args.proxy_clients}
    
    if mode == 'pretrain':
        return model_all, optimizer_all, criterion_all, lr_scheduler_all, wd_scheduler_all, loss_scaler_all
    else:
        return model_all, optimizer_all, criterion_all, lr_scheduler_all, wd_scheduler_all, loss_scaler_all, mixup_fn_all


# def average_model(args, model_avg, model_all):
#     model_avg.cpu()
#     print('Calculate the model avg----')
#     params = dict(model_avg.named_parameters())
        
#     for name, param in params.items():
#         for client in range(len(args.proxy_clients)):
#             single_client = args.proxy_clients[client]
#             single_client_weight = args.clients_weightes[single_client]
#             single_client_weight = torch.from_numpy(np.array(single_client_weight)).float()
            
#             if client == 0:
#                 tmp_param_data = dict(model_all[single_client].module.named_parameters())[
#                                      name].data * single_client_weight
#             else:
#                 tmp_param_data = tmp_param_data + \
#                                  dict(model_all[single_client].module.named_parameters())[
#                                      name].data * single_client_weight
#         params[name].data.copy_(tmp_param_data)
        
#     print('Update each client model parameters----')
    
#     for single_client in args.proxy_clients:
#         tmp_params = dict(model_all[single_client].module.named_parameters())
#         for name, param in params.items():
#             tmp_params[name].data.copy_(param.data)


def average_model(args, model_avg, model_all):
    model_avg.cpu()
    print('Calculate the model avg----')
    params = dict(model_avg.named_parameters())
        
    for name, param in params.items():
        for client in range(len(args.proxy_clients)):
            single_client = args.proxy_clients[client]
            
            single_client_weight = args.clients_weightes[single_client]
            single_client_weight = torch.from_numpy(np.array(single_client_weight)).float()
            
            if client == 0:
                if args.distributed:
                    tmp_param_data = dict(model_all[single_client].module.named_parameters())[
                                         name].data * single_client_weight
                else:
                    tmp_param_data = dict(model_all[single_client].named_parameters())[
                                         name].data * single_client_weight
            else:
                if args.distributed:
                    tmp_param_data = tmp_param_data + \
                                     dict(model_all[single_client].module.named_parameters())[
                                         name].data * single_client_weight
                else:
                    tmp_param_data = tmp_param_data + \
                                     dict(model_all[single_client].named_parameters())[
                                         name].data * single_client_weight
        
        params[name].data.copy_(tmp_param_data)
        
    print('Update each client model parameters----')
    
    # print('model_avg: ', next(model_avg.parameters()).device)
    
    for single_client in args.proxy_clients:
        
        # print('debug: ', dict(model_all[single_client].module.named_parameters()).keys())
        if args.distributed:
            tmp_params = dict(model_all[single_client].module.named_parameters())
        else:
            tmp_params = dict(model_all[single_client].named_parameters())
        
        # print('local_rank: ', args.local_rank)
        # print('model_all device: ', next(model_all[single_client].module.parameters()).device)
            
        for name, param in params.items():
            # print(tmp_params[name].data)
            # print(params[name].data)
            # exit(0)
            tmp_params[name].data.copy_(param.data)