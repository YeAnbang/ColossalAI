SAVE_DIR="/home/yeanbang/data/experiments/sft/colossal-llama2-7b_sft_32k_with_mask"

rm -rf $SAVE_DIR/cache
rm -rf $SAVE_DIR/jsonl
rm -rf $SAVE_DIR/arrow

python prepare_dataset_no_mask.py --type sft \
    --data_input_dirs /home/yeanbang/data/dataset/13b_sft_data \
    --conversation_template_config /home/yeanbang/data/ColossalAI/applications/ColossalChat/config/conversation_template/colossal-llama2.json \
    --tokenizer_dir  "/home/yeanbang/data/models/colossal-llama2-chat-7b" \
    --data_cache_dir $SAVE_DIR/cache \
    --data_jsonl_output_dir $SAVE_DIR/jsonl \
    --data_arrow_output_dir $SAVE_DIR/arrow \