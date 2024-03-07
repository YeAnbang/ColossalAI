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

# export NCCL_IB_HCA=mlx5_1:1,mlx5_2:1,mlx5_3:1,mlx5_4:1
# export NCCL_IB_DISABLE=0
# export NCCL_SOCKET_IFNAME=ens

# export CUDA_VISIBLE_DEVICES=4,5,6
set_n_least_used_CUDA_VISIBLE_DEVICES 8
PROJECT_NAME="activation_beacon"
PARENT_SAVE_DIR="/mnt/jfs/yeanbang/experiments/cont_pretrain/colossal-llama2-7b_cont_pretrain_32k_activation_beacon/ckpt" # Path to a folder to save checkpoints
PARENT_TENSORBOARD_DIR="/mnt/jfs/yeanbang/experiments/cont_pretrain/colossal-llama2-7b_cont_pretrain_32k_activation_beacon/log" # Path to a folder to save logs
PARENT_CONFIG_FILE="/mnt/jfs/yeanbang/experiments/cont_pretrain/colossal-llama2-7b_cont_pretrain_32k_activation_beacon/config" # Path to a folder to save training config logs
PRETRAINED_MODEL_PATH="/mnt/jfs/yeanbang/colossal-llama2-chat-7b" # huggingface or local model path
PRETRAINED_TOKENIZER_PATH="/mnt/jfs/yeanbang/colossal-llama2-chat-7b" # huggingface or local tokenizer path
declare -a original_dataset=(
    /mnt/jfs/yeanbang/dataset_tokenized/colossal-llama2-7b_8k_cont_pretrain_original/arrow/part-00000
    /mnt/jfs/yeanbang/dataset_tokenized/colossal-llama2-7b_8k_cont_pretrain_original/arrow/part-00001
    /mnt/jfs/yeanbang/dataset_tokenized/colossal-llama2-7b_8k_cont_pretrain_original/arrow/part-00002
    /mnt/jfs/yeanbang/dataset_tokenized/colossal-llama2-7b_8k_cont_pretrain_original/arrow/part-00003
    /mnt/jfs/yeanbang/dataset_tokenized/colossal-llama2-7b_8k_cont_pretrain_original/arrow/part-00004
    /mnt/jfs/yeanbang/dataset_tokenized/colossal-llama2-7b_8k_cont_pretrain_original/arrow/part-00005
    /mnt/jfs/yeanbang/dataset_tokenized/colossal-llama2-7b_8k_cont_pretrain_original/arrow/part-00006
    /mnt/jfs/yeanbang/dataset_tokenized/colossal-llama2-7b_8k_cont_pretrain_original/arrow/part-00007
    /mnt/jfs/yeanbang/dataset_tokenized/colossal-llama2-7b_8k_cont_pretrain_original/arrow/part-00008
    /mnt/jfs/yeanbang/dataset_tokenized/colossal-llama2-7b_8k_cont_pretrain_original/arrow/part-00009
) 

declare -a augment_dataset=(
    /mnt/jfs/yeanbang/dataset_tokenized/colossal-llama2-7b_8k_cont_pretrain_augment/arrow/part-00000
    /mnt/jfs/yeanbang/dataset_tokenized/colossal-llama2-7b_8k_cont_pretrain_augment/arrow/part-00001
    /mnt/jfs/yeanbang/dataset_tokenized/colossal-llama2-7b_8k_cont_pretrain_augment/arrow/part-00002
    /mnt/jfs/yeanbang/dataset_tokenized/colossal-llama2-7b_8k_cont_pretrain_augment/arrow/part-00003
    /mnt/jfs/yeanbang/dataset_tokenized/colossal-llama2-7b_8k_cont_pretrain_augment/arrow/part-00004
    /mnt/jfs/yeanbang/dataset_tokenized/colossal-llama2-7b_8k_cont_pretrain_augment/arrow/part-00005
    /mnt/jfs/yeanbang/dataset_tokenized/colossal-llama2-7b_8k_cont_pretrain_augment/arrow/part-00006
    /mnt/jfs/yeanbang/dataset_tokenized/colossal-llama2-7b_8k_cont_pretrain_augment/arrow/part-00007
    /mnt/jfs/yeanbang/dataset_tokenized/colossal-llama2-7b_8k_cont_pretrain_augment/arrow/part-00008
    /mnt/jfs/yeanbang/dataset_tokenized/colossal-llama2-7b_8k_cont_pretrain_augment/arrow/part-00009
)

TIMESTAMP=$(date +%Y-%m-%d-%H-%M-%S)
FULL_PROJECT_NAME="${PROJECT_NAME}-${TIMESTAMP}"
SAVE_DIR="${PARENT_SAVE_DIR}${FULL_PROJECT_NAME}"
CONFIG_FILE="${PARENT_CONFIG_FILE}-${FULL_PROJECT_NAME}.json"

# the real batch size for gradient descent is number_of_node_in_hostfile * nproc_per_node * train_batch_size
colossalai run --nproc_per_node 8 --master_port 31343 train.py \
    --pretrain $PRETRAINED_MODEL_PATH \
    --tokenizer_dir $PRETRAINED_TOKENIZER_PATH \
    --save_interval 1000 \
    --original_dataset ${original_dataset[@]} \
    --augment_dataset ${augment_dataset[@]} \
    --save_path $SAVE_DIR \
    --config_file $CONFIG_FILE \
    --lora_rank 0 \
    --plugin zero2 \
    --batch_size 1 \
    --max_epochs 1 \
    --accumulation_steps 1 \
    --lr 5e-5 \
    --max_len 8192 \
    --grad_checkpoint \
    --use_wandb
