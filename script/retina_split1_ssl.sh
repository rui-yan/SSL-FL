#!/bin/bash

cd /home/yan/SSL-FL/unilm/beit/

DATASET='Retina'
SPLIT_TYPE='split_1'
N_CLASSES=2
DATA_PATH="/data/yan/SSL-FL/${DATASET}/"

# ------------------ pretrain ----------------- #
# EPOCHS=500
EPOCHS=2000
LR='5e-4'
OUTPUT_PATH="/data/yan/SSL-FL/fedavg_model_ckpt/${DATASET}_pretrained_beit_base/pretrained_epoch${EPOCHS}_${SPLIT_TYPE}_lr${LR}"

CUDA_VISIBLE_DEVICES=0 python run_beit_pretraining_FedAvg.py \
        --data_path ${DATA_PATH} \
        --data_set ${DATASET} \
        --output_dir ${OUTPUT_PATH} \
        --lr ${LR} \
        --batch_size 64 \
        --save_ckpt_freq 20 \
        --max_communication_rounds ${EPOCHS} \
        --split_type ${SPLIT_TYPE} \
        --num_mask_patches 75 \
        --model beit_base_patch16_224_8k_vocab \
        --discrete_vae_weight_path /data/yan/SSL-FL/tokenizer_weight \
        --warmup_epochs 10 \
        --clip_grad 3.0 --drop_path 0.1 --layer_scale_init_value 0.1 \
        --E_epoch 1  --num_local_clients -1 \

# ------------------ finetune ----------------- #
CKPT_PATH="${OUTPUT_PATH}/checkpoint-1999.pth"
LR='5e-4'
EPOCHS=100
OUTPUT_PATH_FT="/data/yan/SSL-FL/fedavg_model_ckpt/${DATASET}_pretrained_beit_base/finetune_${DATASET}_epoch${EPOCHS}_${SPLIT_TYPE}_lr${LR}"

CUDA_VISIBLE_DEVICES=0 python run_class_finetuning_FedAvg.py \
     --data_path ${DATA_PATH} \
     --data_set ${DATASET} \
     --finetune ${CKPT_PATH} \
     --nb_classes ${N_CLASSES} \
     --output_dir ${OUTPUT_PATH_FT} \
     --lr ${LR} \
     --save_ckpt_freq 10 \
     --disable_eval_during_finetuning \
     --model beit_base_patch16_224 \
     --batch_size 64 --update_freq 1 --split_type ${SPLIT_TYPE} \
     --warmup_epochs 5 --layer_decay 0.65 --drop_path 0.2 \
     --weight_decay 0.05 --layer_scale_init_value 0.1 --clip_grad 3.0 \
     --E_epoch 1 --max_communication_rounds 100 --num_local_clients -1 
        
        
# ------------------ evaluate ----------------- #
CKPT_PATH="${OUTPUT_PATH_FT}/checkpoint-best.pth"
CUDA_VISIBLE_DEVICES=0 python run_class_finetuning_FedAvg.py \
    --eval --model beit_base_patch16_224 --data_path $DATA_PATH \
    --nb_classes ${N_CLASSES} --data_set ${DATASET} \
    --resume $CKPT_PATH \
    --batch_size 64 \
    --E_epoch 1 --max_communication_rounds 100 --num_local_clients -1 --split_type ${SPLIT_TYPE}
