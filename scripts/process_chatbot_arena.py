import json
import torch
import argparse
import os
import sys
import pandas as pd
from transformers import LlamaTokenizerFast, LlamaForCausalLM
from vllm import LLM, SamplingParams
import numpy as np

def pad(list, n):
    return list + [0.] * (n - len(list))

def prepare_for_arrow(tensor, type):
    return tensor.cpu().numpy().astype(type).tobytes()

def main(args):
    dataset_in = args.dataset_in
    dataset_out = args.dataset_out

    print("Loading dataset...")
    dataset = pd.read_parquet(dataset_in)

    mask = dataset['model'] == args.target_model
    filtered_dataset = dataset[mask]
    filtered_dataset_iter = filtered_dataset.iterrows()

    print(f"found {len(filtered_dataset)} conversations for {args.target_model}")

    print("Loading models...")
    # pad on the left side to make it easier to slice out final hidden states
    tokenizer = LlamaTokenizerFast.from_pretrained(
        args.main_model, 
        max_length=args.prompt_tokens + args.generation_tokens,
        padding_side='left',
        truncation_side='left')
    tokenizer.pad_token = tokenizer.eos_token
    
    main_model = LlamaForCausalLM.from_pretrained(args.main_model, use_cache=False, torch_dtype=torch.float16, device_map="auto")
    draft_model = LlamaForCausalLM.from_pretrained(args.draft_model, use_cache=False, torch_dtype=torch.float16)
    main_model.half()
    draft_model.cuda()

    missed_tokens = 0

    print("Processing dataset...")
    dataset_processed = []

    def save():
        nonlocal dataset_processed
        nonlocal dataset_out
        # save as dataframe
        print(f"saving dataset to {dataset_out}")
        df = pd.DataFrame(
            dataset_processed, 
            columns=["input_ids", "main_hidden_states", "draft_hidden_states", "accept_mask", "dataset_index"])
        # save as parquet
        # append to the existing dataset
        if os.path.exists(dataset_out):
            df.to_parquet(dataset_out, engine='fastparquet', append=True)
        else:
            df.to_parquet(dataset_out, engine='fastparquet')
        dataset_processed = []

    total_processed = 0
    end_reached = False

    while total_processed < args.n and not end_reached:
        batch = []
        while len(batch) < args.batch_size:
            try:
                index, item = next(filtered_dataset_iter)

                conversation = item['conversation']

                state = 0
                prompt = ""
                generation = ""
                
                for c in conversation:
                    if state == 0 and c['role'] == 'user':
                        prompt = c['content']
                        state = 1
                    if state == 1 and c['role'] == 'assistant':
                        generation = c['content']
                        break

                generation_ids = tokenizer.encode(generation)
                if len(generation_ids) < args.generation_tokens:
                    continue

                bi = {
                    "text": prompt + " " + generation,
                    "index": index,
                }
                batch.append(bi)
            except StopIteration:
                end_reached = True
                break

        # prepare batch for forward pass
        prompts = [b['text'] for b in batch]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
        input_ids = inputs.input_ids
        attn_mask = inputs.attention_mask
        input_ids = input_ids[:, -(args.prompt_tokens+args.generation_tokens):]
        attn_mask = attn_mask[:, -(args.prompt_tokens+args.generation_tokens):]

        # run a normal forward pass to get hidden states
        main_hidden_states = main_model.get_decoder().forward(
            input_ids.cuda(),
            attention_mask=attn_mask.cuda(),
        )[0]
        # get hidden states
        main_hidden_states = main_hidden_states[:, -(args.generation_tokens+1):-1, :]
        
        draft_hidden_states = draft_model.get_decoder().forward(
            input_ids.cuda(),
            attention_mask=attn_mask.cuda(),
            output_hidden_states=True,
        )[0]
        draft_hidden_states = draft_hidden_states[:, -(args.generation_tokens+1):-1, :]

        draft_logits = draft_model.get_output_embeddings().forward(draft_hidden_states)
        draft_prediction = torch.argmax(draft_logits, dim=-1).cpu()
        generation_ids = input_ids[:,-args.generation_tokens:]
        # get mask of tokens where the draft model was incorrect
        mask = torch.where(draft_prediction != generation_ids, 1, 0)

        valid = 0
        for i, _ in enumerate(batch):
            # drop it if not the max length
            missed_tokens += mask[i].sum().item()
            seq_input_ids = input_ids[i]
            seq_input_ids = seq_input_ids[seq_input_ids != 2]
            __import__('pdb').set_trace()
            item = {
                "input_ids": prepare_for_arrow(seq_input_ids, np.int64),
                "main_hidden_states": prepare_for_arrow(main_hidden_states[i], np.float32),
                "draft_hidden_states": prepare_for_arrow(draft_hidden_states[i], np.float32),
                "accept_mask": prepare_for_arrow(mask[i], np.float32),
                "dataset_index": batch[i]['index'],
            }
            dataset_processed.append(item)
            total_processed += 1
            valid += 1

        print(f"added {valid} items")

        save()
        torch.cuda.empty_cache()

    total_generated = total_processed * args.generation_tokens

    print("Total generated tokens:", total_generated)
    print("Total accepted tokens:", total_generated - missed_tokens)
    print("Acceptance rate:", (total_generated - missed_tokens) / total_generated)

    save()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_in", type=str, help="Path to dataset")
    parser.add_argument("dataset_out", type=str, help="Path to processed dataset")
    parser.add_argument("--prompt_tokens", type=int, default=256)
    parser.add_argument("--generation_tokens", type=int, default=256)
    parser.add_argument("--main_model", type=str, default="JackFram/llama-160m")
    parser.add_argument("--draft_model", type=str, default="JackFram/llama-160m")
    parser.add_argument("--target_model", type=str, default="llama-13b")
    parser.add_argument("--n", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--writeback_interval", type=int, default=1000)
    args = parser.parse_args()

    with torch.no_grad():
        main(args)
