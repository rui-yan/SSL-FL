#!/bin/bash

cd /home/yan/SSL-FL/unilm/beit/

DATASET='COVIDx'
SPLIT_TYPE='split_1'
N_CLASSES=3
DATA_PATH="/raid/yan/SSL-FL/${DATASET}/"

# ------------------ finetune ----------------- #
EPOCHS=100
LR='1.5e-3'
OUTPUT_PATH="/raid/yan/SSL-FL/fedavg_model_ckpt/imnet_pretrained_beit_base/finetune_${DATASET}_epoch${EPOCHS}_${SPLIT_TYPE}_lr${LR}"
CKPT_PATH='/raid/yan/SSL-FL/fedavg_model_ckpt/imnet_pretrained_beit_base/beit_base_patch16_224_pt22k.pth'

CUDA_VISIBLE_DEVICES=3 python run_class_finetuning_FedAvg.py \
     --data_path ${DATA_PATH} \
     --data_set ${DATASET} \
     --finetune ${CKPT_PATH} \
     --nb_classes ${N_CLASSES} \
     --output_dir ${OUTPUT_PATH} \
     --lr ${LR} \
     --save_ckpt_freq 20 \
     --model beit_base_patch16_224 \
     --batch_size 128  --update_freq 1 --split_type ${SPLIT_TYPE} \
     --warmup_epochs 5 --layer_decay 0.65 --drop_path 0.2 \
     --weight_decay 0.05 --layer_scale_init_value 0.1 --clip_grad 3.0 \
     --E_epoch 1 --max_communication_rounds ${EPOCHS} --num_local_clients -1 

# ------------------ evaluate ----------------- #
CKPT_PATH="${OUTPUT_PATH}/checkpoint-best.pth"
CUDA_VISIBLE_DEVICES=3 python run_class_finetuning_FedAvg.py \
    --eval --model beit_base_patch16_224 --data_path $DATA_PATH \
    --nb_classes ${N_CLASSES} --data_set ${DATASET} \
    --resume $CKPT_PATH \
    --batch_size 128 \
    --E_epoch 1 --max_communication_rounds 100 --num_local_clients -1 --split_type ${SPLIT_TYPE}
