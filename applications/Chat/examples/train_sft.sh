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

set_n_least_used_CUDA_VISIBLE_DEVICES 1

# the real batch size for gradient descent is number_of_node_in_hostfile * nproc_per_node * train_batch_size
colossalai run --nproc_per_node 1 --master_port 28537 --hostfile ./hostfile train_sft.py \
    --pretrain "bigscience/bloom-560m" \
    --plugin zero2 \
    --save_path /home/lcyab/data/test_folder/model_checkpoint/gpt2 \
    --dataset tatsu-lab/alpaca \
    --batch_size 4 \
    --max_epochs 1 \
    --max_datasets_size 20000 \
    --accumulation_steps 8 \
    --lr 2e-5 \
    --lora_rank 30 \
    --lora_rank 30 \
    --max_len 512 \
    --max_epochs 1 \
    --use_flash_attn \
    --use_wandb
    # --grad_checkpoint \
