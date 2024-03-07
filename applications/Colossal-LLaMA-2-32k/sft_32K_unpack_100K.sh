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
PROJECT_NAME="sft_200K"
PARENT_SAVE_DIR="/home/yeanbang/data/experiments/cont_pretrain/colossal-llama2-7b_sft_unpack_200K_v1/ckpt" # Path to a folder to save checkpoints
PARENT_TENSORBOARD_DIR="/home/yeanbang/data/experiments/cont_pretrain/colossal-llama2-7b_sft_unpack_200K_v1/log" # Path to a folder to save logs
PARENT_CONFIG_FILE="/home/yeanbang/data/experiments/cont_pretrain/colossal-llama2-7b_sft_unpack_200K_v1/config" # Path to a folder to save training config logs
PRETRAINED_MODEL_PATH="/home/yeanbang/data/experiments/cont_pretrain/colossal-llama2-7b_sft_unpack_v1/ckptsft_32K-2024-03-04-16-24-07/epoch-0_step-2000/modeling" # huggingface or local model path
PRETRAINED_TOKENIZER_PATH="/home/zhongyuting/model/Colossal-LLaMA-2-7b-base" # huggingface or local tokenizer path
DATA_SOURCE="/home/yeanbang/jfs/ColossalAI/applications/Colossal-LLaMA-2-32k/sft_100K_dataset_unpack_sft_long_alpaca.json"

TIMESTAMP=$(date +%Y-%m-%d-%H-%M-%S)
FULL_PROJECT_NAME="${PROJECT_NAME}-${TIMESTAMP}"
SAVE_DIR="${PARENT_SAVE_DIR}${FULL_PROJECT_NAME}"
CONFIG_FILE="${PARENT_CONFIG_FILE}-${FULL_PROJECT_NAME}.json"

# the real batch size for gradient descent is number_of_node_in_hostfile * nproc_per_node * train_batch_size
# colossalai run --nproc_per_node 8 --master_port 31343 --hostfile hostfile cont_pretrain.py \
# colossalai run --nproc_per_node 8 --master_port 31344 cont_pretrain_32K_new.py \
# torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=10.20.1.171 --master_port=31344 cont_pretrain_32K_new.py \
# colossalai run --nproc_per_node 8 --master_port 31344 cont_pretrain_32K_new.py \
# torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=10.20.1.82 --master_port=31344 cont_pretrain_32K_new.py \
# colossalai run --nproc_per_node 8 --master_port 31344 cont_pretrain_32K_new.py \
# torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=10.20.1.179 --master_port=31344 cont_pretrain_32K_new.py \
colossalai run --nproc_per_node 8 --master_port 31344 cont_pretrain_32K_new.py \
    --pretrain $PRETRAINED_MODEL_PATH \
    --tokenizer_dir $PRETRAINED_TOKENIZER_PATH \
    --save_interval 200 \
    --cont_pretrain_dataset_config $DATA_SOURCE \
    --save_path $SAVE_DIR \
    --config_file $CONFIG_FILE \
    --scaling_type dynamic \
    --scaling_factor 4 \
    --plugin 3d \
    --tp 8 \
    --batch_size 1 \
    --max_epochs 3 \
    --accumulation_steps 12 \
    --lr 1e-5 \
    --max_len 102400 \
    --grad_checkpoint \
    --use_flash_attn 
