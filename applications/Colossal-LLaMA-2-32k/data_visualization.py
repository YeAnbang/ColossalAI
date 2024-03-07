import matplotlib.pyplot as plt
import json

data_config = json.load(open('/home/yeanbang/jfs/ColossalAI/applications/Colossal-LLaMA-2-32k/cont_pretrain_32K_dataset_config_original_sft_full.json', 'r', encoding='utf8'))
data_name = 'sft_original'
cpt_stat = {}
sft_stat = {}
for dataset_name in data_config:
    ratio = data_config[dataset_name]['to_keep_ratio']
    total_samples = data_config[dataset_name]['total_samples']
    total_input_tokens = data_config[dataset_name]['total_input_tokens']
    total_label_tokens = data_config[dataset_name]['total_label_tokens']
    type = data_config[dataset_name]['type']
    if type=='sft':
        target = sft_stat
    else:
        target = cpt_stat
    target[dataset_name] = {
        'total_samples': int(total_samples*ratio),
        'total_input_tokens': int(total_input_tokens*ratio),
        'total_label_tokens': int(total_label_tokens*ratio)
    }

# Visualization
for data_split_name, data_split in zip(['cpt', 'sft'], [cpt_stat, sft_stat]):
    if len(data_split)!=0:
        plt.figure(figsize=(50, 5))
        plt.title(f'{data_split_name.upper()} Number of Samples per Category')
        plt.bar(data_split.keys(), [v['total_samples'] for v in data_split.values()])
        plt.xlabel('Category')
        plt.ylabel('Total samples')
        plt.savefig(f'./data_statistics/{data_name}_number_samples_per_category.png')

        plt.figure(figsize=(50, 5))
        plt.title(f'{data_split_name.upper()} Number of Input Tokens Per Category')
        plt.bar(data_split.keys(), [v['total_input_tokens'] for v in data_split.values()])
        plt.xlabel('Category')
        plt.ylabel('Total input tokens')
        plt.savefig(f'./data_statistics/{data_name}_number_input_tokens_per_category.png')

        plt.figure(figsize=(50, 5))
        plt.title(f'{data_split_name.upper()} Number of Label Tokens Per Category')
        plt.bar(data_split.keys(), [v['total_label_tokens'] for v in data_split.values()])
        plt.xlabel('Category')
        plt.ylabel('Total label tokens')
        plt.savefig(f'./data_statistics/{data_name}_number_label_tokens_per_category.png')
    
        print(f"Estimate total samples in {data_name} dataset:", sum([v['total_samples'] for v in data_split.values()]))
        print(f"Estimate total input tokens in {data_name} dataset:", sum([v['total_input_tokens'] for v in data_split.values()]))
        print(f"Estimate total label tokens in {data_name} dataset:", sum([v['total_label_tokens'] for v in data_split.values()]))