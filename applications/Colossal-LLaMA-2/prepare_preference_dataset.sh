rm -rf /home/lcyab/data/data_rlhf/tokenized_preference_data/cache
rm -rf /home/lcyab/data/data_rlhf/tokenized_preference_data/jsonl
rm -rf /home/lcyab/data/data_rlhf/tokenized_preference_data/arrow

python prepare_preference_dataset.py --data_input_dirs /home/lcyab/data/data_rlhf/preprcessed \
    --pretrained  "/mnt/vepfs/lcxyc/leaderboard_models/Colossal-LLaMA-2-7b-base/" \
    --data_cache_dir /home/lcyab/data/data_rlhf/tokenized_preference_data/cache \
    --data_jsonl_output_dir /home/lcyab/data/data_rlhf/tokenized_preference_data/jsonl \
    --data_arrow_output_dir /home/lcyab/data/data_rlhf/tokenized_preference_data/arrow