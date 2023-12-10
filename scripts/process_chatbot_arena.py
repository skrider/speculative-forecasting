import json
import os
import sys
import pandas as pd
from transformers import LlamaTokenizerFast

MAX_TOKENS = 256

def main(args):
    dataset_in = args[0]
    dataset_out = args[1]

    with open(dataset_in, "r") as f:
        dataset = json.load(f)

    tokenizer = LlamaTokenizerFast.from_pretrained("JackFram/llama-160m")

    dataset_processed = []
    for d in dataset:
        acc = ""
        state = 0
        for c in d['conversations']:
            if state == 0 and c['from'] == 'human':
                acc = acc + ' ' + c['value']
                state = 1
            if state == 1 and c['from'] == 'gpt':
                acc = acc + ' ' + c['value']
                state = 2
            if state == 2 and c['from'] == 'human':
                acc = acc + ' ' + c['value']
                break
        input_ids = tokenizer.encode(acc, return_tensors="pt").squeeze(0)
        if input_ids.shape[0] > MAX_TOKENS:
            input_ids = input_ids[:MAX_TOKENS]
        acc = tokenizer.decode(input_ids)
        item = {
            "text": acc,
            "tokens": input_ids.tolist()
        }

        dataset_processed.append(item)
    # save as dataframe
    df = pd.DataFrame(dataset_processed, columns=['text', 'tokens'])
    # save as parquet
    df.to_parquet(dataset_out)

if __name__ == "__main__":
    main(sys.argv[1:])
