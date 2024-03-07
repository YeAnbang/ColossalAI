import argparse
import json
import math
import os
import resource
from contextlib import nullcontext
from datasets import load_from_disk
from datasets import Dataset as HFDataset
import torch
from coati.dataset import (
    DataCollatorForSupervisedDataset,
    load_tokenized_dataset,
    setup_conversation_template,
    setup_distributed_dataloader,
)
from coati.models import convert_to_lora_module
from coati.trainer import SFTTrainer
from coati.utils import load_checkpoint

from llama_rope_utils import LlamaForCausalLM, LlamaConfig
from transformers import AutoTokenizer
import random
import colossalai
from colossalai.booster import Booster
from colossalai.booster.plugin import (
    GeminiPlugin, 
    HybridParallelPlugin, 
    LowLevelZeroPlugin, 
    TorchDDPPlugin
)
from colossalai.cluster import DistCoordinator
from colossalai.lazy import LazyInitContext
from colossalai.nn.lr_scheduler import LambdaLR
from colossalai.nn.optimizer import HybridAdam
from colossalai.utils import get_current_device

from dynamic_ntk import patch_llama_for_dynamic_scaled_rotary_embeddings

def train(args):
    # check lora compatibility
    if "gemini" in args.plugin:
        if args.lora_rank > 0:
            raise ValueError("LoRA is not supported in GeminiPlugin. Please use other plugin")
        if args.accumulation_steps > 1:
            raise ValueError("Gradient accumulation is not supported in GeminiPlugin. Please use other plugin")
    # ==============================
    # Initialize Distributed Training
    # ==============================
    colossalai.launch_from_torch({})
    coordinator = DistCoordinator()

    # ==============================
    # Initialize Booster
    # ==============================
    if args.plugin == "ddp":
        '''
        Default torch ddp plugin without any acceleration, for 
        debugging purpose acceleration, for debugging purpose
        '''
        plugin = TorchDDPPlugin(find_unused_parameters=True)
    elif args.plugin == "gemini":
        plugin = GeminiPlugin(
            precision=args.mixed_precision,
            placement_policy="static",
            initial_scale=2**16,
            max_norm=args.grad_clip,
        )
    elif args.plugin == "gemini_auto":
        plugin = GeminiPlugin(
            precision=args.mixed_precision,
            placement_policy="auto",
            initial_scale=2**16,
            max_norm=args.grad_clip,
        )
    elif args.plugin == "zero2":
        plugin = LowLevelZeroPlugin(
            stage=2,
            precision=args.mixed_precision,
            initial_scale=2**16,
            max_norm=args.grad_clip,
        )
    elif args.plugin == "zero2_cpu":
        plugin = LowLevelZeroPlugin(
            stage=2,
            precision=args.mixed_precision,
            initial_scale=2**16,
            cpu_offload=True,
            max_norm=args.grad_clip,
        )
    elif args.plugin == "3d":
        plugin = HybridParallelPlugin(
            tp_size=args.tp,
            pp_size=1,
            zero_stage=0,
            max_norm=args.grad_clip,
            precision=args.mixed_precision,
        )
    else:
        raise ValueError(f"Unknown plugin {args.plugin}")

    booster = Booster(plugin=plugin)
    original_max_position_embeddings=4096

    model_config = LlamaConfig.from_pretrained(args.pretrain)
    if args.max_position_embeddings is None:
        model_config.max_position_embeddings = int(args.scaling_factor * original_max_position_embeddings)
    else:
        model_config.max_position_embeddings = args.max_position_embeddings
    model_config.rope_theta = args.rope_theta
    model_config.rope_scaling = {
        "type": args.scaling_type,
        "factor": args.scaling_factor,
        "original_max_position_embeddings": original_max_position_embeddings
    }

    # ======================================================
    # Initialize Model, Objective, Optimizer and LR Scheduler
    # ======================================================
    # Temp Fix: Disable lazy init due to version conflict
    # init_ctx = (
    #     LazyInitContext(default_device=get_current_device()) if isinstance(plugin, (GeminiPlugin,)) else nullcontext()
    # )
    init_ctx = nullcontext()
    with init_ctx:
        if args.use_flash_attn:
            model = LlamaForCausalLM.from_pretrained(args.pretrain, 
                        torch_dtype=torch.bfloat16 if args.mixed_precision=='bf16' else torch.float16, 
                        config = model_config,
                        use_flash_attention_2=True)
            # patch_llama_for_dynamic_scaled_rotary_embeddings(model, ntk=False)
            coordinator.print_on_master(msg="Flash-attention enabled successfully")
        else:
            model = LlamaForCausalLM.from_pretrained(args.pretrain, 
                        torch_dtype=torch.bfloat16 if args.mixed_precision=='bf16' else torch.float16, 
                        config = model_config,
                        use_flash_attention_2=False)
            # patch_llama_for_dynamic_scaled_rotary_embeddings(model, ntk=False)
        if args.lora_rank > 0:
            model = convert_to_lora_module(model, args.lora_rank, lora_train_bias=args.lora_train_bias)

    if args.grad_checkpoint and args.lora_rank == 0:
        # lora layers are not supported by gradient checkpointing
        model.gradient_checkpointing_enable()
        coordinator.print_on_master(msg="Gradient checkpointing enabled successfully")
    elif args.lora_rank > 0:
        coordinator.print_on_master(msg="Gradient checkpointing will be disabled when LoRA is enabled")


    # configure tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir or args.pretrain, use_fast=False, trust_remote_code=True)
    if hasattr(tokenizer, 'pad_token') and hasattr(tokenizer, 'eos_token') and tokenizer.eos_token is not None:
        try:
            # Some tokenizers doesn't allow to set pad_token mannually e.g., Qwen
           tokenizer.pad_token = tokenizer.eos_token
        except AttributeError as e:
            logger.warning(f"Unable to set pad token to eos token, {str(e)}")
    if not hasattr(tokenizer, 'pad_token') or tokenizer.pad_token is None:
        logger.warning("The tokenizer does not have a pad token which is required. May lead to unintended behavior in training, Please consider manually set them.")

    tokenizer.add_bos_token = False
    tokenizer.add_eos_token = False

    coordinator.print_on_master(f"Configuration file will be saved at: {args.config_file}")
    coordinator.print_on_master(f"Model checkpoint will be saved at: {args.save_path}")

    # configure optimizer
    optim = HybridAdam(
        model_params=model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
        adamw_mode=True,
    )

    # configure dataset
    coordinator.print_on_master(
        f"Max CUDA memory before data loader: {torch.cuda.max_memory_allocated() / 1024 ** 2:.2f} MB"
    )
    # 48043 samples
    coordinator.print_on_master(f"start loading datasets")
    datasets = []  # `List[datasets.dataset_dict.Dataset]`
    dataset_length = 0
    for ds_path in args.augment_cont_pretrain_dataset:
        ds_path = os.path.abspath(ds_path)
        assert os.path.exists(ds_path), f"Not existed file path {ds_path}"
        ds_dict = load_from_disk(dataset_path=ds_path, keep_in_memory=False)
        if isinstance(ds_dict, HFDataset):
            datasets.append(ds_dict)
            dataset_length += len(ds_dict)
        else:
            raise ValueError("Only hf dataset is allowed")
    coordinator.print_on_master("augment_cont_pretrain_dataset loaded")
    coordinator.print_on_master(f"Total size: {dataset_length}")
    
    # 255674 samples, take 100000 samples
    dataset_length = 0
    ratio = 100000/255674
    for ds_path in args.original_cont_pretrain_dataset:
        ds_path = os.path.abspath(ds_path)
        assert os.path.exists(ds_path), f"Not existed file path {ds_path}"
        ds_dict = load_from_disk(dataset_path=ds_path, keep_in_memory=False)
        if isinstance(ds_dict, HFDataset):
            ds_dict = ds_dict.select(random.sample([i for i in range(len(ds_dict))], int(len(ds_dict)*ratio)))
            datasets.append(ds_dict)
            dataset_length += len(ds_dict)
        else:
            raise ValueError("Only hf dataset is allowed")
    coordinator.print_on_master("original_cont_pretrain_dataset loaded")
    coordinator.print_on_master(f"Total size: {dataset_length}")

    # 348991 samples, take 50000 samples
    ratio = 50000/348991
    dataset_length = 0
    for ds_path in args.original_sft_dataset:
        ds_path = os.path.abspath(ds_path)
        assert os.path.exists(ds_path), f"Not existed file path {ds_path}"
        ds_dict = load_from_disk(dataset_path=ds_path, keep_in_memory=False)
        if isinstance(ds_dict, HFDataset):
            ds_dict = ds_dict.select(random.sample([i for i in range(len(ds_dict))], int(len(ds_dict)*ratio)))
            datasets.append(ds_dict)
            dataset_length += len(ds_dict)
        else:
            raise ValueError("Only hf dataset is allowed")
    coordinator.print_on_master("original_sft_dataset loaded")
    coordinator.print_on_master(f"Total size: {dataset_length}")

    from torch.utils.data import ConcatDataset
    dataset = ConcatDataset(datasets=datasets)

    coordinator.print_on_master(f"Merged dataset size: {len(dataset)}")

    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer, max_length=args.max_len)
    # set random seed
    torch.manual_seed(47)
    train_dataloader = setup_distributed_dataloader(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=data_collator,
        use_tp=args.tp > 1,
    )
    # # log training data for debuging
    dtype = None
    import torch.distributed as dist
    if not dist.is_initialized() or dist.get_rank() == 0:
        with open('./debug.jsonl', 'w', encoding='utf8') as f_out:
            for i, batch in enumerate(train_dataloader):
                for i in range(batch['input_ids'].shape[0]):
                    dtype = batch['input_ids'].dtype
                    f_out.write(json.dumps({'input_ids':batch['input_ids'][i].tolist(), "labels":batch['labels'][i].tolist()}, ensure_ascii=False)+'\n')

    coordinator.print_on_master(
        f"Max CUDA memory after data loader: {torch.cuda.max_memory_allocated() / 1024 ** 2:.2f} MB"
    )

    num_update_steps_per_epoch = len(train_dataloader) // args.accumulation_steps
    math.ceil(args.max_epochs * num_update_steps_per_epoch)

    if args.warmup_steps is None:
        args.warmup_steps = int(args.max_epochs * 0.025 * (len(train_dataloader) // args.accumulation_steps))
        coordinator.print_on_master(f"Warmup steps is set to {args.warmup_steps}")

    lr_scheduler = LambdaLR(
        optimizer=optim,
        total_steps=args.max_epochs * num_update_steps_per_epoch,
        lr_lambda=lambda step: 1.0
    )

    # Flash attention will be disabled because it does NOT support fp32.
    default_dtype = torch.float16 if args.mixed_precision == "fp16" else torch.bfloat16
    torch.set_default_dtype(default_dtype)
    model, optim, _, train_dataloader, lr_scheduler = booster.boost(
        model=model,
        optimizer=optim,
        lr_scheduler=lr_scheduler,
        dataloader=train_dataloader,
    )
    # model = model.to(get_current_device())
    torch.set_default_dtype(torch.float)

    coordinator.print_on_master(f"Booster init max CUDA memory: {torch.cuda.max_memory_allocated() / 1024 ** 2:.2f} MB")
    coordinator.print_on_master(
        f"Booster init max CPU memory: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024:.2f} MB"
    )

    from collections import defaultdict
    import tqdm

    model_ = model.unwrap()

    data_source = "./debug.jsonl"
    prompt = []
    for line in open(data_source, 'r', encoding='utf8'):
        if line=="":
            continue
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
    model_.eval()
    with torch.no_grad():
        with open('./long_context_experiment/evaluation_results_before_fitting.jsonl', 'w', encoding='utf8') as f:
            for each in tqdm.tqdm(prompt):
                input_ids_decode = tokenizer.decode(each['input_ids'], skip_special_tokens=False)
                target_decode = tokenizer.decode(each['target'], skip_special_tokens=False)
                output = model_.generate(input_ids=torch.tensor([each['input_ids']]).to(dtype).to(model_.device), max_new_tokens=2, do_sample=False)
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
    if not dist.is_initialized() or dist.get_rank() == 0:
        print("Before training.")
        for pos in total:
            correct_ += correct[pos]
            total_ += total[pos]
            print(f"Accuracy at pos {pos}000: ", correct[pos]/total[pos])
        print(f"Average Accuracy: ", correct_/total_)

    model.train()

    start_epoch = 0
    sampler_start_idx = 0
    start_step = 0
    if args.checkpoint_path is not None:
        if "modeling" in args.checkpoint_path:
            coordinator.print_on_master(f"Continued pretrain from checkpoint {args.checkpoint_path}")
            booster.load_model(model, args.checkpoint_path)
        else:
            coordinator.print_on_master(f"Load model checkpoint from {args.checkpoint_path}")
            start_epoch, start_step, sampler_start_idx = load_checkpoint(
                load_dir=args.checkpoint_path,
                booster=booster,
                model=model,
                optimizer=optim,
                lr_scheduler=lr_scheduler,
            )
            train_dataloader.sampler.set_start_index(start_index=sampler_start_idx)

            coordinator.print_on_master(
                f"Loaded checkpoint {args.checkpoint_path} at epoch {start_epoch} step {start_step}"
            )
            coordinator.print_on_master(f"Loaded sample at index {sampler_start_idx}")

        coordinator.print_on_master(
            f"Checkpoint loaded max CUDA memory: {torch.cuda.max_memory_allocated() / 1024 ** 2:.2f} MB"
        )
        coordinator.print_on_master(
            f"Checkpoint loaded CUDA memory: {torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB"
        )
        coordinator.print_on_master(
            f"Checkpoint loaded max CPU memory: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024:.2f} MB"
        )

    trainer = SFTTrainer(
        model=model,
        booster=booster,
        optim=optim,
        lr_scheduler=lr_scheduler,
        max_epochs=args.max_epochs,
        accumulation_steps=args.accumulation_steps,
        start_epoch=start_epoch,
        save_interval=args.save_interval,
        save_dir=args.save_path,
        coordinator=coordinator,
    )

    trainer.fit(
        train_dataloader=train_dataloader,
        eval_dataloader=None,
        log_dir=args.log_dir,
        use_wandb=args.use_wandb,
    )

    from collections import defaultdict
    import tqdm

    model = model.unwrap()

    data_source = "./debug.jsonl"
    prompt = []
    for line in open(data_source, 'r', encoding='utf8'):
        if line=="":
            continue
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
        with open('./long_context_experiment/evaluation_results_after_fitting.jsonl', 'w', encoding='utf8') as f:
            for each in tqdm.tqdm(prompt):
                input_ids_decode = tokenizer.decode(each['input_ids'], skip_special_tokens=False)
                target_decode = tokenizer.decode(each['target'], skip_special_tokens=False)
                output = model.generate(input_ids=torch.tensor([each['input_ids']]).to(dtype).to(model.device), max_new_tokens=2, do_sample=False)
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
    import torch.distributed as dist
    if not dist.is_initialized() or dist.get_rank() == 0:
        print("After training.")
        for pos in total:
            correct_ += correct[pos]
            total_ += total[pos]
            print(f"Accuracy at pos {pos}000: ", correct[pos]/total[pos])
        print(f"Average Accuracy: ", correct_/total_)



if __name__ == "__main__":
    # ==============================
    # Parse Arguments
    # ==============================
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--plugin",
        type=str,
        default="gemini",
        choices=["gemini", "gemini_auto", "zero2", "zero2_cpu", "3d", "ddp"],
        help="Choose which plugin to use",
    )
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping value")
    parser.add_argument("--weight_decay", type=float, default=0.1, help="Weight decay")
    parser.add_argument("--warmup_steps", type=int, default=None, help="Warmup steps")
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--pretrain", type=str, default=None)
    parser.add_argument("--tokenizer_dir", type=str, default=None)
    parser.add_argument("--original_cont_pretrain_dataset", nargs="*", default=[])
    parser.add_argument("--augment_cont_pretrain_dataset", nargs="*", default=[])
    parser.add_argument("--original_sft_dataset", nargs="*", default=[])
    parser.add_argument(
        "--checkpoint_path", type=str, default=None, help="Checkpoint path if need to resume training form a checkpoint"
    )
    parser.add_argument("--save_path", type=str, default="output")
    parser.add_argument("--max_epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_len", type=int, default=32*1024)
    parser.add_argument("--scaling_type", type=str, default='linear', choices=['linear', 'dynamic', 'yarn', 'dynamic-yarn'])
    parser.add_argument("--scaling_factor", type=int, default=8)
    parser.add_argument("--rope_theta", type=float, default=10000.0)

    parser.add_argument("--original_max_position_embeddings ", type=int, default=4096)
    parser.add_argument("--max_position_embeddings", type=int, default=None)
    parser.add_argument("--mixed_precision", type=str, default="fp16", choices=["fp16", "bf16"], help="Mixed precision")
    parser.add_argument("--lora_rank", type=int, default=0, help="low-rank adaptation matrices rank")
    parser.add_argument(
        "--lora_train_bias",
        type=str,
        default="none",
        help="'none' means it doesn't train biases. 'all' means it trains all biases. 'lora_only' means it only trains biases of LoRA layers",
    )
    parser.add_argument("--save_interval", type=int, default=1000, help="number of step between two checkpoints")
    parser.add_argument("--merge_lora_weights", type=bool, default=True)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--config_file", type=str, default="config_file", help="Config file")
    parser.add_argument("--accumulation_steps", type=int, default=8)
    parser.add_argument("--log_dir", default="logs", type=str)
    parser.add_argument("--use_wandb", default=False, action="store_true")
    parser.add_argument("--grad_checkpoint", default=False, action="store_true")
    parser.add_argument("--use_flash_attn", default=False, action="store_true")
    args = parser.parse_args()
    os.makedirs(os.path.dirname(args.config_file), exist_ok=True)
    with open(args.config_file, "w") as f:
        json.dump(args.__dict__, f, indent=4)
    train(args)
