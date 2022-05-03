#!/bin/bash

MODEL_NAME='beit'

cd /home/yan/SSL-FL/${MODEL_NAME}/

DATASET='ISIC'
SPLIT_TYPE='central'
N_CLASSES=7
DATA_PATH="/data/yan/SSL-FL/${DATASET}/"
N_CLIENTS=5
MASK_RATIO=0.4

# ------------------ pretrain ----------------- #--
EPOCHS=1000
LR='2e-3'
BATCH_SIZE=64

OUTPUT_PATH="/data/yan/SSL-FL/fedavg_${MODEL_NAME}_ckpt_${N_CLIENTS}/${DATASET}_pretrained_beit_base/pretrained_epoch${EPOCHS}_${SPLIT_TYPE}_lr${LR}_bs${BATCH_SIZE}_ratio${MASK_RATIO}_dis4"

CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=4 run_beit_pretrain_FedAvg.py \
        --data_path ${DATA_PATH} \
        --data_set ${DATASET} \
        --output_dir ${OUTPUT_PATH} \
        --lr ${LR} \
        --batch_size ${BATCH_SIZE} \
        --save_ckpt_freq 50 \
        --max_communication_rounds ${EPOCHS} \
        --split_type ${SPLIT_TYPE} \
        --num_mask_patches 75 \
        --model beit_base_patch16_224_8k_vocab \
        --discrete_vae_weight_path /data/yan/SSL-FL/tokenizer_weight \
        --warmup_epochs 10 --sync_bn \
        --clip_grad 3.0 --drop_path 0.1 --layer_scale_init_value 0.1 \
        --n_clients ${N_CLIENTS} --E_epoch 1  --num_local_clients -1 \

# ------------------ finetune ----------------- #
CKPT_PATH="${OUTPUT_PATH}/checkpoint-999.pth"
FT_EPOCHS=100
FT_LR='3e-3'
FT_BATCH_SIZE=64
OUTPUT_PATH_FT="${OUTPUT_PATH}/finetune_${DATASET}_epoch${FT_EPOCHS}_${SPLIT_TYPE}_lr${FT_LR}_bs${FT_BATCH_SIZE}_dis4"

CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=4 runc_class_finetune_FedAvg.py \
     --data_path ${DATA_PATH} \
     --data_set ${DATASET} \
     --finetune ${CKPT_PATH} \
     --nb_classes ${N_CLASSES} \
     --output_dir ${OUTPUT_PATH_FT} \
     --lr ${FT_LR} \
     --save_ckpt_freq 50 \
     --model beit_base_patch16_224 \
     --batch_size ${FT_BATCH_SIZE} --update_freq 1 --split_type ${SPLIT_TYPE} \
     --warmup_epochs 5 --layer_decay 0.65 --drop_path 0.2 --sync_bn \
     --weight_decay 0.05 --layer_scale_init_value 0.1 --clip_grad 3.0 \
     --n_clients ${N_CLIENTS} --E_epoch 1 --max_communication_rounds ${FT_EPOCHS} --num_local_clients -1 

