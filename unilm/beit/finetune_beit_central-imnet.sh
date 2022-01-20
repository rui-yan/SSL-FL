#!/bin/bash

# Set the path to save checkpoints
# OUTPUT_DIR='/data/yan/SSL-FL/model_ckpt/cifar10_pretrained_beit_base/finetune_cifar10_epoch90'
# OUTPUT_DIR='/data/yan/SSL-FL/fedavg_model_ckpt/random_init_beit_base/finetune_cifar10_epoch600_central'
# OUTPUT_DIR='/raid/yan/SSL-FL/fedavg_model_ckpt/cifar10_pretrained_beit_base/finetune_cifar10_epoch600_central_itercorrected'
# OUTPUT_DIR='/raid/yan/SSL-FL/fedavg_model_ckpt/imnet_pretrained_beit_base/finetune_retina_epoch200_central_lr3e-3'
OUTPUT_DIR='/data/yan/SSL_FL/fedavg_model_ckpt/imnet_pretrained_beit_base/covidx_finetune_epoch90_central_1.5e-3'

# Download and extract CIFAR10
# DATA_PATH='/data/yan/cifar10'
# DATA_PATH='/raid/yan/Retina'
DATA_PATH='/data/yan/SSL_FL/COVIDx'

# Set the path to saved checkpoints to resume
# CKPT_PATH='/home/yan/SSL-FL/checkpoints/cifar10_pretrained_beit_base/finetune_cifar10_epoch10_lr1e-3/checkpoint-9.pth'
# CKPT_PATH='/data/yan/SSL-FL/model_ckpt/random_init_beit_base/finetune_cifar10_epoch400/checkpoint-399.pth'
# CKPT_PATH='/data/yan/SSL-FL/fedavg_model_ckpt/cifar10_pretrained_beit_base/pretrained_epoch400_5e-3_spilt_1/checkpoint-399.pth'
# CKPT_PATH='/data/yan/SSL-FL/fedavg_model_ckpt/cifar10_pretrained_beit_base/pretrained_epoch400_5e-3_central/checkpoint-399.pth'
# CKPT_PATH='/raid/yan/SSL-FL/fedavg_model_ckpt/imnet_pretrained_beit_base/beit_base_patch16_224_pt22k.pth'

CUDA_VISIBLE_DEVICES=6 OMP_NUM_THREADS=1 python run_class_finetuning_FedAvg.py \
     --model beit_base_patch16_224 --data_path $DATA_PATH \
     --finetune https://unilm.blob.core.windows.net/beit/beit_base_patch16_224_pt22k.pth \
     --nb_classes 3 --data_set COVIDx --disable_eval_during_finetuning \
     --output_dir $OUTPUT_DIR --batch_size 32 --lr 1.5e-3 --update_freq 1 \
     --warmup_epochs 5 --layer_decay 0.65 --drop_path 0.2 \
     --weight_decay 0.05 --layer_scale_init_value 0.1 --clip_grad 3.0 \
     --E_epoch 1 --max_communication_rounds 90 --num_local_clients -1 --split_type central


# --finetune https://unilm.blob.core.windows.net/beit/beit_base_patch16_224_pt22k.pth \
# CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 python run_class_finetuning_FedAvg.py \
#      --model beit_base_patch16_224 --data_path $DATA_PATH \
#      --nb_classes 10 --data_set CIFAR10 --disable_eval_during_finetuning \
#      --output_dir $OUTPUT_DIR --batch_size 32 --lr 3e-3 --update_freq 1 \
#      --warmup_epochs 5 --layer_decay 0.65 --drop_path 0.2 \
#      --weight_decay 0.05 --layer_scale_init_value 0.1 --clip_grad 3.0 \
#      --E_epoch 1 --max_communication_rounds 500 --num_local_clients -1 --split_type central


#      --finetune $CKPT_PATH \ 
# finetune 
# OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=8 run_class_finetuning.py \
#      --model beit_base_patch16_224 --data_path $DATA_PATH \
#      --nb_classes 10 --data_set CIFAR10 --disable_eval_during_finetuning \
#      --output_dir $OUTPUT_DIR --batch_size 32 --lr 3e-3 --update_freq 1 \
#      --warmup_epochs 5 --epochs 90 --layer_decay 0.65 --drop_path 0.2 \
#      --weight_decay 0.05 --layer_scale_init_value 0.1 --clip_grad 3.0

# OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=8 run_class_finetuning.py \
#      --model beit_base_patch16_224 --data_path $DATA_PATH \
#      --nb_classes 10 --data_set CIFAR10 --disable_eval_during_finetuning \
#      --output_dir $OUTPUT_DIR --batch_size 32 --lr 3e-3 --update_freq 1 \
#      --warmup_epochs 5 --epochs 600 --layer_decay 0.65 --drop_path 0.2 \
#      --weight_decay 0.05 --layer_scale_init_value 0.1 --clip_grad 3.0
     
# --finetune $CKPT_PRETRAINED_PATH \
# --finetune https://unilm.blob.core.windows.net/beit/beit_base_patch16_224_pt22k.pth \


# evaluate
# CUDA_VISIBLE_DEVICES=3 python run_class_finetuning.py \
#     --eval --model beit_base_patch16_224 --data_path $DATA_PATH \
#     --nb_classes 10 --data_set CIFAR10 \
#     --resume $CKPT_PATH \
#     --batch_size 32
    # --max_communication_rounds 100 --num_local_clients -1 --split_type split_1 --E_epoch 1
