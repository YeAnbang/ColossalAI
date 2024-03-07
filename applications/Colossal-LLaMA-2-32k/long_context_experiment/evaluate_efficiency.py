from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import tqdm
import torch 
from collections import defaultdict

model = AutoModelForCausalLM.from_pretrained("/home/yeanbang/data/models/colossal-llama2-7b-32K/modeling",
                                             use_flash_attention_2=True, torch_dtype=torch.bfloat16, device_map='auto')
tokenizer = AutoTokenizer.from_pretrained("/home/yeanbang/data/models/colossal-llama2-7b-32K/modeling")

data_source = "/home/yeanbang/jfs/ColossalAI/applications/Colossal-LLaMA-2-32k/debug.jsonl"
prompt = []
for line in open(data_source, 'r', encoding='utf8'):
    data = json.loads(line)
    input_ids = data['input_ids']
    labels = data['labels']
    start = 0
    for i in tqdm.tqdm(range(len(input_ids))):
        if input_ids[i]==1:
            # sequence start
            if start != 0:
                prompt.append({})
                for j in range(start, i):
                    if labels[j]!=-100:
                        prompt[-1]["target"] = labels[j]
                        prompt[-1]["input_ids"] = input_ids[start:j]
                        prompt[-1]['start']=start
                        prompt[-1]['end']=j-1
                        break
            start = i

evaluation_results = []
total = defaultdict(lambda : 0)
correct = defaultdict(lambda : 0)
model.eval()
with torch.no_grad():
    with open('evaluation_results.jsonl', 'w', encoding='utf8') as f:
        for each in tqdm.tqdm(prompt):
            input_ids_decode = tokenizer.decode(each['input_ids'], skip_special_tokens=False)
            target_decode = tokenizer.decode(each['target'], skip_special_tokens=False)
            output = model.generate(input_ids=torch.tensor([each['input_ids']]).to(torch.long).to('cuda'), max_new_tokens=2, do_sample=False)
            new_tokens = output[0][len(each['input_ids']):]
            result = tokenizer.decode(new_tokens, skip_special_tokens=True)
            # print(input_ids_decode,'#', target_decode, '#', result)
            correctness = new_tokens[0].item()==each['target']
            pos = (each['start']+each['end'])/2
            total[int(pos/2000)] += 1
            if correctness:
                correct[int(pos/2000)] += 1
            evaluation_results.append({"source": input_ids_decode, "target": target_decode, "output": result, "correctness": correctness, "start": each['start'], "end": each['end']})
            f.write(json.dumps({"source": input_ids_decode, "target": target_decode, "output": result, "correctness": correctness, "start": each['start'], "end": each['end']}, ensure_ascii=False)+'\n')
correct_ = 0
total_ = 0
for pos in total:
    correct_ += correct[pos]
    total_ += total[pos]
    print(f"Accuracy at pos {pos}000: ", correct[pos]/total[pos])
print(f"Average Accuracy: ", correct_/total_)