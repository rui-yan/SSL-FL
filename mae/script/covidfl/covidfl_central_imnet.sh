#!/bin/bash

MODEL_NAME='mae'

cd /home/yan/SSL-FL/${MODEL_NAME}/

DATASET='COVIDfl'
SPLIT_TYPE='central'
N_CLASSES=3
DATA_PATH="/data/yan/SSL-FL/${DATASET}/"
N_CLIENTS=12
N_GPUS=4

# ------------------ finetune ----------------- #
CKPT_PATH='/data/yan/SSL-FL/mae_ckpt/mae_pretrain_vit_base.pth'

FT_EPOCHS=100
FT_LR='3e-3'
FT_BATCH_SIZE=16

OUTPUT_PATH_FT="/data/yan/SSL-FL/fedavg_${MODEL_NAME}_ckpt_${N_CLIENTS}/imnet_pretrain_${MODEL_NAME}_base/finetune_${DATASET}_epoch${FT_EPOCHS}_${SPLIT_TYPE}_lr${FT_LR}_bs${FT_BATCH_SIZE}_dis${N_GPUS}"

CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=${N_GPUS} run_class_finetune_FedAvg.py \
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
