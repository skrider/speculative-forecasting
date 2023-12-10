import json
import os
import sys
import pandas as pd

def main(args):
    dataset_in = args[0]
    dataset_out = args[1]

    with open(dataset_in, "r") as f:
        dataset = json.load(f)

    dataset_processed = []
    for d in dataset:
        acc = ""
        state = 0
        for c in d['conversations']:
            if state == 0 and c['from'] == 'human':
                acc = acc + ' ' + c['value']
                state = 1
            if state == 1 and c['from'] == 'bot':
                acc = acc + ' ' + c['value']
                state = 2
            if state == 2 and c['from'] == 'human':
                acc = acc + ' ' + c['value']
                break
        dataset_processed.append(acc)
    # save as dataframe
    df = pd.DataFrame(dataset_processed, columns=['text'])
    # save as parquet
    df.to_parquet(dataset_out)

if __name__ == "__main__":
    main(sys.argv[1:])
