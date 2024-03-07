SAVE_DIR="/home/yeanbang/data/dataset/failed_case_in_general_eval"

rm -rf $SAVE_DIR/cache
rm -rf $SAVE_DIR/jsonl
rm -rf $SAVE_DIR/arrow

python prepare_continue_pretrain_dataset.py \
    --data_input_dirs /home/yeanbang/data/dataset/failed_case_in_general_eval \
    --tokenizer_dir /mnt/jfs/yeanbang/Colossal-LLaMA-2-7b-base \
    --data_cache_dir $SAVE_DIR/cache \
    --data_jsonl_output_dir $SAVE_DIR/jsonl \
    --data_arrow_output_dir $SAVE_DIR/arrow \
    --max_length 32768
