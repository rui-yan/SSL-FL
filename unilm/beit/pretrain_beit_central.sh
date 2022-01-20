#!/bin/bash

# Set the path to save checkpoints
# OUTPUT_DIR='/home/yan/SSL-FL/model_ckpt/cifar10_pretrained_beit_base/pretrained_epoch400'
OUTPUT_DIR='/data/yan/SSL_FL/fedavg_model_ckpt/covidx_pretrained_beit_base/pretrained_epoch400_central_lr1.5e-3'

# Download and extract CIFAR10
# DATA_PATH='/raid/yan/cifar10'
# DATA_PATH='/data/yan/Retina/
DATA_PATH='/data/yan/SSL_FL/COVIDx'

# Download the tokenizer weight from OpenAI's DALL-E
TOKENIZER_PATH='/data/yan/SSL_FL/tokenizer_weight'
# mkdir -p $TOKENIZER_PATH
# wget -o $TOKENIZER_PATH/encoder.pkl https://cdn.openai.com/dall-e/encoder.pkl
# wget -o $TOKENIZER_PATH/decoder.pkl https://cdn.openai.com/dall-e/decoder.pkl

# OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=8 run_beit_pretraining_FedAvg.py \
#         --data_path ${DATA_PATH} --data_set CIFAR10 --output_dir ${OUTPUT_DIR} --num_mask_patches 75 \
#         --model beit_base_patch16_224_8k_vocab --discrete_vae_weight_path ${TOKENIZER_PATH} \
#         --batch_size 32 --lr 5e-3 --warmup_epochs 10 \
#         --clip_grad 3.0 --drop_path 0.1 --layer_scale_init_value 0.1 \
#         --E_epoch 1 --max_communication_rounds 100 --num_local_clients -1 --split_type split_2

CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=1 python run_beit_pretraining_FedAvg.py \
--data_path ${DATA_PATH} --data_set COVIDx --output_dir ${OUTPUT_DIR} --num_mask_patches 75 \
--model beit_base_patch16_224_8k_vocab --discrete_vae_weight_path ${TOKENIZER_PATH} \
--batch_size 32 --lr 1.5e-3 --warmup_epochs 10 --save_ckpt_freq 20 \
--clip_grad 3.0 --drop_path 0.1 --layer_scale_init_value 0.1 \
--E_epoch 1 --max_communication_rounds 400 --num_local_clients -1 --split_type central
