#!/bin/bash

set_n_least_used_CUDA_VISIBLE_DEVICES() {
    local n=${1:-"9999"}
    echo "GPU Memory Usage:"
    local FIRST_N_GPU_IDS=$(nvidia-smi --query-gpu=memory.used --format=csv |
        tail -n +2 |
        nl -v 0 |
        tee /dev/tty |
        sort -g -k 2 |
        awk '{print $1}' |
        head -n $n)
    export CUDA_VISIBLE_DEVICES=$(echo $FIRST_N_GPU_IDS | sed 's/ /,/g')
    echo "Now CUDA_VISIBLE_DEVICES is set to:"
    echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
}
# set_n_least_used_CUDA_VISIBLE_DEVICES 4
export CUDA_VISIBLE_DEVICES=2,3,4,5
# NCCL IB environment variables
export NCCL_IB_HCA=mlx5_1:1,mlx5_2:1,mlx5_3:1,mlx5_4:1
export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_GID_INDEX=3
export NCCL_IB_TIMEOUT=23
export NCCL_IB_RETRY_CNT=7
export OMP_NUM_THREADS=8


PROJECT_NAME="llama2-dpo"
PARENT_SAVE_DIR="./output/ckpt"
PARENT_TENSORBOARD_DIR="./output/tensorboard"
PARENT_CONFIG_FILE="./output/train_config"
PRETRAINED_MODEL_PATH="/mnt/vepfs/lcxyc/leaderboard_models/Colossal-LLaMA-2-7b-base/"

declare -a dataset=(
    /home/lcyab/data/data_rlhf/tokenized_preference_data/arrow/part-00000
    /home/lcyab/data/data_rlhf/tokenized_preference_data/arrow/part-00001
)

TIMESTAMP=$(date +%Y-%m-%d-%H-%M-%S)
FULL_PROJECT_NAME="${PROJECT_NAME}-${TIMESTAMP}"
SAVE_DIR="${PARENT_SAVE_DIR}${FULL_PROJECT_NAME}"
TENSORBOARD_DIR="${PARENT_TENSORBOARD_DIR}${FULL_PROJECT_NAME}"
CONFIG_FILE="${PARENT_CONFIG_FILE}${FULL_PROJECT_NAME}.json"

colossalai run --nproc_per_node 4 --hostfile hostfile --master_port 30014 train_dpo.py \
    --pretrain $PRETRAINED_MODEL_PATH \
    --dataset ${dataset[@]} \
    --plugin "3d" \
    --save_interval 400 \
    --save_dir $SAVE_DIR \
    --max_epochs 1 \
    --accumulation_steps 4 \
    --batch_size 4 \
    --lr 5e-6 \
    --mixed_precision "bf16" \
    --grad_clip 1.0 \
    --weight_decay 0.01 \
    --warmup_steps 100 \
    --grad_checkpoint \
    # --use_wandb
