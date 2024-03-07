from loader import load_mixed_cont_pretrain_dataset_from_config
import os
import json
from pathlib import Path
import glob
import torch
import tqdm
from multiprocessing import cpu_count
from collections import defaultdict
import matplotlib.pyplot as plt

def find_files(directory, extension=None):
    out = []
    start_path = Path(directory)
    jsonl_files = start_path.rglob(extension)
    for file_path in jsonl_files:
        out.append(file_path)
    return out

def mapping_fn(sample):
    ret = {}
    ret['num_tokens_input_ids'] = len(sample['input_ids'])
    ret['num_tokens_label'] = len(sample['labels'])-sample['labels'].count(-100)
    del sample['input_ids']
    del sample['labels']
    return ret

if __name__=='__main__':
    cpt_data_source_dirs = [
                        ]
    sft_data_source_dirs = [
        '/home/yeanbang/data/dataset/augmented_cont_pretrain_dataset/dataset_tokenized/colossal-llama2-7b_32k_cont_pretrain_original_sft_full/*',
    ]
    output_path = './data_statistics'
    output_config_file_name = "cont_pretrain_32K_dataset_config_original_sft_full.json"
    data_source_dirs = cpt_data_source_dirs + sft_data_source_dirs
    empty_config = {}

    for path in data_source_dirs:
        if os.path.exists(path):
            split_name = path.split("/")[-1]
            all_files = find_files(path, extension="*.arrow")
            assert len(all_files) > 0, f"No dataset files (*.arrow) found in {path}"
            folders = sorted(list(set(['/'.join(str(file).split("/")[:-1]) for file in all_files])))
            if split_name not in empty_config:
                empty_config[split_name] = {}
            empty_config[split_name]['tokenized_dataset_dir'] = folders
            empty_config[split_name]['type'] = "cpt" if path in cpt_data_source_dirs else "sft"
            empty_config[split_name]['to_keep_ratio'] = 1.0
        else:
            folders = glob.glob(path)
            assert len(folders) > 0, "The input data source path should either be a directory of a glob expression of directories"
            data_source_dirs.extend(folders)
            if path in cpt_data_source_dirs:
                cpt_data_source_dirs.extend(folders)
            else:
                sft_data_source_dirs.extend(folders)
    for dataset_name in empty_config:
        print(f"Calculate dataset statistics for {dataset_name} dataset...")
        dataset_config = {dataset_name: empty_config[dataset_name]}
        dataset = load_mixed_cont_pretrain_dataset_from_config(dataset_config, mapping = mapping_fn)
        total_sample = len(dataset)
        total_input_tokens = 0 #sum(dataset['num_tokens_input_ids'])
        total_label_tokens = 0 #sum(dataset['num_tokens_label'])
        data_statistics = defaultdict(lambda : 0)
        for i in range(len(dataset.datasets)):
            subset = dataset.datasets[i]
            for j in tqdm.tqdm(range(len(subset))):
                num_tokens_input_ids = subset[j]['num_tokens_input_ids']
                num_tokens_label = subset[j]['num_tokens_label']
                data_statistics[int(num_tokens_input_ids/1000)]+=1
                total_input_tokens += num_tokens_input_ids
                total_label_tokens += num_tokens_label
        
        print(f"Total samples: {total_sample}")
        print(f"Total input tokens: {total_input_tokens}")
        print(f"Total label tokens: {total_label_tokens}")
        empty_config[dataset_name]['total_samples'] = total_sample
        empty_config[dataset_name]['total_input_tokens'] = total_input_tokens
        empty_config[dataset_name]['total_label_tokens'] = total_label_tokens


    with open(output_config_file_name, "w", encoding='utf8') as f:
        json.dump(empty_config, f, indent=4, ensure_ascii=False)

    with open(output_config_file_name, "r", encoding='utf8') as f:
        empty_config = json.load(f) 