#!/bin/bash

MODEL_NAME='mae'

cd /home/yan/SSL-FL/${MODEL_NAME}/

DATASET='ISIC'
SPLIT_TYPE='central'
N_CLASSES=7
DATA_PATH="/data/yan/SSL-FL/${DATASET}/"
N_CLIENTS=5
MASK_RATIO=0.6
AUG='aug_2'

# ------------------ pretrain ----------------- #--
EPOCHS=1600
BLR='1.5e-3'
BATCH_SIZE=32

OUTPUT_PATH="/data/yan/SSL-FL/fedavg_${MODEL_NAME}_ckpt_${N_CLIENTS}/${DATASET}_pretrained_beit_base/pretrained_epoch${EPOCHS}_${SPLIT_TYPE}_blr${BLR}_bs${BATCH_SIZE}_ratio${MASK_RATIO}_dis4_${AUG}"

CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=4 run_mae_pretrain_FedAvg.py \
        --data_path ${DATA_PATH} \
        --data_set ${DATASET} \
        --model_name ${MODEL_NAME} \
        --output_dir ${OUTPUT_PATH} \
        --blr ${BLR} \
        --batch_size ${BATCH_SIZE} \
        --save_ckpt_freq 50 \
        --max_communication_rounds ${EPOCHS} \
        --split_type ${SPLIT_TYPE} \
        --mask_ratio ${MASK_RATIO}\
        --model mae_vit_base_patch16 \
        --warmup_epochs 40 \
        --weight_decay 0.05 \
        --norm_pix_loss --sync_bn \
        --aug ${AUG} \
        --n_clients ${N_CLIENTS} --E_epoch 1  --num_local_clients -1 \

# ------------------ finetune ----------------- #
CKPT_PATH="${OUTPUT_PATH}/checkpoint-1599.pth"
FT_EPOCHS=100
FT_LR='3e-3'
FT_BATCH_SIZE=64
OUTPUT_PATH_FT="${OUTPUT_PATH}/finetune_${DATASET}_epoch${FT_EPOCHS}_${SPLIT_TYPE}_lr${FT_LR}_bs${FT_BATCH_SIZE}_dis"

CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=4 run_class_finetune_FedAvg.py \
     --data_path ${DATA_PATH} \
     --data_set ${DATASET} \
     --model_name ${MODEL_NAME} \
     --finetune ${CKPT_PATH} \
     --nb_classes ${N_CLASSES} \
     --output_dir ${OUTPUT_PATH_FT} \
     --save_ckpt_freq 50 \
     --model vit_base_patch16 \
     --batch_size ${FT_BATCH_SIZE} --split_type ${SPLIT_TYPE} \
     --blr ${FT_LR} --layer_decay 0.65 \
     --weight_decay 0.05 --drop_path 0.1 --reprob 0.25 --mixup 0.8 --cutmix 1.0 \
     --n_clients ${N_CLIENTS} --E_epoch 1 --max_communication_rounds ${FT_EPOCHS} --num_local_clients -1 