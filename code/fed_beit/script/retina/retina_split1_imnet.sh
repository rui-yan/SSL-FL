#!/bin/bash
# This script run finetuning on Retina dataset with imagenet pre-trained model
# Please modify the path DATA_PATH, OUTPUT_PATH

FED_MODEL='fed_beit'

cd /home/yan/SSL-FL/code/${FED_MODEL}/ # change this to "your_path/SSL/code/${FED_MODEL}"

DATASET='Retina' # dataset name
DATA_PATH="/home/yan/SSL-FL/data/${DATASET}/" # change DATA_PATH to the path where the data were stored

SPLIT_TYPE='split_1' # chosen from {'split_1', 'split_2', 'split_3'}
N_CLASSES=2 # the number of classes in the dataset
N_CLIENTS=5 # number of clients in the federated setting
N_GPUS=4 # the number of GPUs used for model training

# ------------------ finetune ----------------- #
CKPT_PATH='/home/yan/SSL-FL/data/ckpts/beit_base_patch16_224_pt22k.pth'

FT_EPOCHS=100
FT_LR='3e-3'
FT_BATCH_SIZE=64

# change OUTPUT_PATH_FT to your path where the pre-trained checkpoints will be stored
OUTPUT_PATH_FT="/home/yan/SSL-FL/data/ckpts/${DATASET}/imnet_pretrain/finetune_${DATASET}_epoch${FT_EPOCHS}_${SPLIT_TYPE}_lr${FT_LR}_bs${FT_BATCH_SIZE}_dis${N_GPUS}"

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
     --warmup_epochs 5 --layer_decay 0.65 --drop_path 0.2 \
     --weight_decay 0.05 --layer_scale_init_value 0.1 --clip_grad 3.0 \
     --n_clients ${N_CLIENTS} --E_epoch 1 --max_communication_rounds ${FT_EPOCHS} --num_local_clients -1
     
# # ------------------ evaluate ----------------- #
# CKPT_PATH="${OUTPUT_PATH_FT}/checkpoint-best.pth"
# CUDA_VISIBLE_DEVICES=0 python run_class_finetune_FedAvg.py \
#     --eval --model beit_base_patch16_224 --data_path $DATA_PATH \
#     --nb_classes ${N_CLASSES} --data_set ${DATASET} \
#     --resume $CKPT_PATH \
#     --batch_size ${FT_BATCH_SIZE} \
#     --E_epoch 1 --max_communication_rounds 100 --num_local_clients -1 --split_type ${SPLIT_TYPE}