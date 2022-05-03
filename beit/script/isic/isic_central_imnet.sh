#!/bin/bash

MODEL_NAME='beit'

cd /home/cihangxie/yan/SSL-FL/unilm/${MODEL_NAME}/

DATASET='ISIC'
SPLIT_TYPE='central'
N_CLASSES=2
DATA_PATH="/data1/data/yan/SSL-FL/${DATASET}/"
N_CLIENTS=5

# ------------------ finetune ----------------- #
CKPT_PATH='/data1/data/yan/SSL-FL/beit_ckpt/beit_base_patch16_224_pt22k.pth'

FT_EPOCHS=100
FT_LR='3e-3'
FT_BATCH_SIZE=32

OUTPUT_PATH_FT="/data1/data/yan/SSL-FL/fedavg_${MODEL_NAME}_ckpt_${N_CLIENTS}/imnet_pretrained_${MODEL_NAME}_base/finetune_${DATASET}_epoch${FT_EPOCHS}_${SPLIT_TYPE}_lr${FT_LR}_bs${FT_BATCH_SIZE}_dis8"

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=8 main_finetune.py \
     --data_path ${DATA_PATH} \
     --data_set ${DATASET} \
     --finetune ${CKPT_PATH} \
     --nb_classes ${N_CLASSES} \
     --output_dir ${OUTPUT_PATH_FT} \
     --lr ${FT_LR} \
     --save_ckpt_freq 50 \
     --model beit_base_patch16_224 \
     --batch_size ${FT_BATCH_SIZE} --update_freq 1 --split_type ${SPLIT_TYPE} \
     --warmup_epochs 5 --layer_decay 0.65 --drop_path 0.2 \
     --weight_decay 0.05 --layer_scale_init_value 0.1 --clip_grad 3.0 \
     --n_clients ${N_CLIENTS} --E_epoch 1 --max_communication_rounds ${FT_EPOCHS} --num_local_clients -1
