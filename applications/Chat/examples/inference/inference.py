import argparse
import os
from copy import deepcopy

import torch
from chatio import dummy_io, rich_io, simple_io
from coati.dataset.conversation import default_conversation
from coati.models import generate_streaming
from transformers import AutoModelForCausalLM, AutoTokenizer


def get_gpu_memory(max_gpus=None):
    gpu_memory = []
    num_gpus = torch.cuda.device_count() if max_gpus is None else min(max_gpus, torch.cuda.device_count())

    for gpu_id in range(num_gpus):
        with torch.cuda.device(gpu_id):
            device = torch.cuda.current_device()
            gpu_properties = torch.cuda.get_device_properties(device)
            total_memory = gpu_properties.total_memory / (1024**3)
            allocated_memory = torch.cuda.memory_allocated() / (1024**3)
            available_memory = total_memory - allocated_memory
            gpu_memory.append(available_memory)
    return gpu_memory


def load_model_and_tokenizer(model_path, tokenizer_path, device="cuda", **kwargs):
    model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    tokenizer.pad_token = tokenizer.eos_token
    model.to(device)

    return model, tokenizer


def generation_wrapper(*args, **kwargs):
    input_ids = args[1]
    tokenizer = args[2]
    for output in generate_streaming(*args, **kwargs):
        yield tokenizer.batch_decode(output[:, input_ids.size(1) :], skip_special_tokens=True)[0]


def main(args):
    max_new_tokens = args.max_new_tokens
    model_max_length = args.model_max_length
    model, tokenizer = load_model_and_tokenizer(
        args.model_path, args.tokenizer_path or args.model_path, local_files_only=True
    )

    assert max_new_tokens <= model_max_length
    if not tokenizer.eos_token_id:
        tokenizer.eos_token_id = "</s>"
    tokenizer.padding_side = "left"

    model_kwargs = {
        "max_new_tokens": max_new_tokens,
        # 'early_stopping': True,
        # 'top_k': -1,
        # 'top_p': 1.0,
        # 'temperature': 1.0,
        # 'temperature':0.1,
    }
    conv = deepcopy(default_conversation)

    roles = conv.roles
    round = 1

    while True:
        if args.io == "simple":
            chat_io = simple_io
        elif args.io == "rich":
            chat_io = rich_io
        elif args.io == "dummy":
            chat_io = dummy_io
        else:
            raise ValueError(f"Unknown io type: {args.io}")
        # raw_text = print(">>> Human:", end=" ")
        inp = chat_io.prompt_for_input(conv.roles[0])

        if not inp:
            print("prompt should not be empty!")
            continue

        if inp.strip() == "clear":
            conv.clear()
            os.system("clear")
            continue

        if inp.strip() == "exit":
            print("End of chat.")
            break

        query_text = inp.strip()

        conv.append_message(roles[0], query_text)
        conv.append_message(roles[1], None)

        chat_io.prompt_for_output(conv.roles[1])

        prompt = conv.get_prompt()
        input_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"].to(
            torch.cuda.current_device()
        )
        output_stream = generation_wrapper(
            model,
            input_ids,
            tokenizer,
            max_length=model_max_length,
            temperature=0.7,
            early_stopping=True,
            **model_kwargs,
        )

        # print(f">>> Assistant:", end=" ")
        outputs = chat_io.stream_output(output_stream)

        conv.messages[-1][-1] = outputs.strip()

        with open("round.txt", mode="a", encoding="utf-8") as f:
            f.write("\n\n" + "=" * 10 + "\n")
            f.write(f"round {round}:\n{conv.save_prompt()}\n\n")
            f.write("=" * 10 + "\n")

        # print(f">>> Assistant:", end=" ")

        round += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--tokenizer_path", type=str, default=None)
    parser.add_argument("--model_max_length", type=int, default=2048)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--io", type=str, default="rich", choices=["simple", "rich", "dummy"])
    args = parser.parse_args()
    main(args)
