#!/bin/bash
# This script run Fed-BEiT pre-training on Retina Split-1 dataset, then finetune
# Please modify the path DATA_PATH, VAE_WEIGHT_PATH, OUTPUT_PATH

FED_MODEL='fed_beit'

cd /home/yan/SSL-FL/code/${FED_MODEL}/ # change this to "your_path/SSL/code/${FED_MODEL}"

DATASET='Retina' # dataset name
DATA_PATH="/home/yan/SSL-FL/data/${DATASET}/" # change DATA_PATH to the path where the data were stored
VAE_WEIGHT_PATH="/home/yan/SSL-FL/data/tokenizer_weight" # change VAE_WEIGHT_PATH to the path where the vae weights were stored

SPLIT_TYPE='split_1' # chosen from {'split_1', 'split_2', 'split_3'}
N_CLASSES=2 # the number of classes in the dataset
N_CLIENTS=5 # number of clients in the federated setting
MASK_RATIO=0.5 # masking ratio for Fed-BEiT pre-training
N_GPUS=4 # the number of GPUs used for model training

# ------------------ Fed-BEiT pretraining ----------------- #
# you can directly use the saved pre-trained checkpoints from our github and skip this step

EPOCHS=1000
LR='2e-3'
BATCH_SIZE=64
# change OUTPUT_PATH to your path where the pre-trained checkpoints will be stored
OUTPUT_PATH="/home/yan/SSL-FL/data/ckpts/${DATASET}/${FED_MODEL}/pretrained_epoch${EPOCHS}_${SPLIT_TYPE}_lr${LR}_bs${BATCH_SIZE}_ratio${MASK_RATIO}_dis${N_GPUS}" 

# change the CUDA devices available for model training 
CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=${N_GPUS} run_beit_pretrain_FedAvg.py \
        --data_path ${DATA_PATH} \
        --data_set ${DATASET} \
        --output_dir ${OUTPUT_PATH} \
        --lr ${LR} \
        --batch_size ${BATCH_SIZE} \
        --save_ckpt_freq 50 \
        --max_communication_rounds ${EPOCHS} \
        --split_type ${SPLIT_TYPE} \
        --mask_ratio ${MASK_RATIO} \
        --model beit_base_patch16_224_8k_vocab \
        --discrete_vae_weight_path ${VAE_WEIGHT_PATH} \
        --warmup_epochs 10 --sync_bn \
        --clip_grad 3.0 --drop_path 0.1 \
        --layer_scale_init_value 0.1 \
        --n_clients ${N_CLIENTS} --E_epoch 1  --num_local_clients -1 \

# ------------------ finetune ----------------- #
CKPT_PATH="${OUTPUT_PATH}/checkpoint-999.pth" # change this path to the path where the pre-trained ckpt was stored 
FT_EPOCHS=100
FT_LR='3e-3'
FT_BATCH_SIZE=64
OUTPUT_PATH_FT="${OUTPUT_PATH}/finetune_${DATASET}_epoch${FT_EPOCHS}_${SPLIT_TYPE}_lr${FT_LR}_bs${FT_BATCH_SIZE}_dis${N_GPUS}"

# change the CUDA devices available for model training 
CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=${N_GPUS} run_class_finetune_FedAvg.py \
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