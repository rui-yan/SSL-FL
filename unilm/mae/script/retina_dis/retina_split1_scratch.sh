#!/bin/bash

MODEL_NAME='mae'

cd /home/yan/SSL-FL/unilm/${MODEL_NAME}/

DATASET='Retina'
SPLIT_TYPE='split_1'
N_CLASSES=2
DATA_PATH="/data/yan/SSL-FL/${DATASET}/"
N_CLIENTS=5

# ------------------ finetune ----------------- #
FT_EPOCHS=1600
FT_LR='3e-3'
FT_BATCH_SIZE=64
OUTPUT_PATH_FT="/data/yan/SSL-FL/fedavg_${MODEL_NAME}_ckpt_${N_CLIENTS}/random_init_beit_base/finetune_${DATASET}_epoch${FT_EPOCHS}_${SPLIT_TYPE}_lr${FT_LR}_bs${FT_BATCH_SIZE}_dis"

CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=4 run_class_finetuning_FedAvg.py \
     --data_path ${DATA_PATH} \
     --data_set ${DATASET} \
     --model_name ${MODEL_NAME} \
     --nb_classes ${N_CLASSES} \
     --output_dir ${OUTPUT_PATH_FT} \
     --save_ckpt_freq 50 \
     --model vit_base_patch16 \
     --batch_size ${FT_BATCH_SIZE} --split_type ${SPLIT_TYPE} \
     --blr ${FT_LR} --layer_decay 0.65 \
     --weight_decay 0.05 --drop_path 0.1 --reprob 0.25 --mixup 0.8 --cutmix 1.0 \
     --n_clients ${N_CLIENTS} --E_epoch 1 --max_communication_rounds ${FT_EPOCHS} --num_local_clients -1 

# CUDA_VISIBLE_DEVICES=3 python run_class_finetuning_FedAvg.py \
#      --data_path ${DATA_PATH} \
#      --data_set ${DATASET} \
#      --nb_classes ${N_CLASSES} \
#      --output_dir ${OUTPUT_PATH} \
#      --lr ${LR} \
#      --save_ckpt_freq 50 \
#      --model beit_base_patch16_224 \
#      --batch_size 128  --update_freq 1 --split_type ${SPLIT_TYPE} \
#      --warmup_epochs 5 --layer_decay 0.65 --drop_path 0.2 \
#      --weight_decay 0.05 --layer_scale_init_value 0.1 --clip_grad 3.0 \
#      --E_epoch 1 --max_communication_rounds ${EPOCHS} --num_local_clients -1 

# # ------------------ evaluate ----------------- #
# CKPT_PATH="${OUTPUT_PATH}/checkpoint-best.pth"
# CUDA_VISIBLE_DEVICES=3 python run_class_finetuning_FedAvg.py \
#     --eval --model beit_base_patch16_224 --data_path $DATA_PATH \
#     --nb_classes ${N_CLASSES} --data_set ${DATASET} \
#     --resume $CKPT_PATH \
#     --batch_size 128 \
#     --E_epoch 1 --max_communication_rounds 100 --num_local_clients -1 --split_type ${SPLIT_TYPE}
