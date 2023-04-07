#!/bin/bash#!/bin/bash
# This script run Fed-MAE pre-training on centralized COVIDFL dataset, then finetune
# Please modify the path DATA_PATH, OUTPUT_PATH

FED_MODEL='fed_mae'

cd /home/yan/SSL-FL/code/${FED_MODEL}/ # change this to "your_path/SSL/code/${FED_MODEL}"

DATASET='COVID-FL' # dataset name
DATA_PATH="/home/yan/SSL-FL/data/${DATASET}/" # change DATA_PATH to the path where the data were stored

SPLIT_TYPE='central' # chosen from {'central', 'split_real'}
N_CLASSES=3 # the number of classes in the dataset
N_CLIENTS=12 # number of clients in the federated setting
MASK_RATIO=0.4 # masking ratio for Fed-MAE pre-training
N_GPUS=4 # the number of GPUs used for model training

# ------------------ Fed-MAE pretraining ----------------- #
# you can directly use the saved pre-trained checkpoints from our github and skip this step

EPOCHS=1600
BLR='1.5e-3'
BATCH_SIZE=16
# change OUTPUT_PATH to your path where the pre-trained checkpoints will be stored
OUTPUT_PATH="/home/yan/SSL-FL/data/ckpts/${DATASET}/${FED_MODEL}/pretrained_epoch${EPOCHS}_${SPLIT_TYPE}_blr${BLR}_bs${BATCH_SIZE}_ratio${MASK_RATIO}_dis${N_GPUS}"

# change the CUDA devices available for model training
CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=${N_GPUS} run_mae_pretrain_FedAvg.py \
        --data_path ${DATA_PATH} \
        --data_set ${DATASET} \
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
        --n_clients ${N_CLIENTS} --E_epoch 1  --num_local_clients -1 \

# ------------------ finetune ----------------- #
CKPT_PATH="${OUTPUT_PATH}/checkpoint-1599.pth"

# Uncomment this line if you want to directly fine-tune from the saved pre-trained checkpoints
CKPT_PATH="/home/yan/SSL-FL/data/ckpts/COVID-FL/covidfl_pretrain_mae_base_central_checkpoint-1599.pth"

FT_EPOCHS=100
FT_LR='3e-3'
FT_BATCH_SIZE=24
OUTPUT_PATH_FT="${OUTPUT_PATH}/finetune_${DATASET}_epoch${FT_EPOCHS}_${SPLIT_TYPE}_lr${FT_LR}_bs${FT_BATCH_SIZE}_dis${N_GPUS}"

# change the CUDA devices available for model training 
CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=${N_GPUS} run_class_finetune_FedAvg.py \
     --data_path ${DATA_PATH} \
     --data_set ${DATASET} \
     --finetune ${CKPT_PATH} \
     --nb_classes ${N_CLASSES} \
     --output_dir ${OUTPUT_PATH_FT} \
     --save_ckpt_freq 50 \
     --model vit_base_patch16 \
     --batch_size ${FT_BATCH_SIZE} --split_type ${SPLIT_TYPE} \
     --blr ${FT_LR} --layer_decay 0.65 \
     --weight_decay 0.05 --drop_path 0.1 --reprob 0.25 --mixup 0.8 --cutmix 1.0 \
     --n_clients ${N_CLIENTS} --E_epoch 1 --max_communication_rounds ${FT_EPOCHS} --num_local_clients -1