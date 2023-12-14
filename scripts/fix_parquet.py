import pandas as pd
import numpy as np
import argparse
import json

def buf_to_array_bytes(buf, type):
    buf_a = json.loads(buf.decode("utf-8"))
    buf_v = np.array(buf_a).astype(type)
    return buf_v.tobytes()

def main(args):
    print("loading dataset")
    df = pd.read_parquet(args.dataset_in, engine="pyarrow")
    df.to_parquet(args.dataset_out, engine="pyarrow")

    new_dataset_acc = []
    for _, row in df.iterrows():
        print(f"processing {row.dataset_index}")
        new_dataset_acc.append({
            "input_ids": buf_to_array_bytes(row.input_ids, np.int64),
            "main_hidden_states": buf_to_array_bytes(row.main_hidden_states, np.float32),
            "draft_hidden_states": buf_to_array_bytes(row.draft_hidden_states, np.float32),
            "accept_mask": buf_to_array_bytes(row.accept_mask, np.int32),
            "dataset_index": row.dataset_index,
        })
    
    new_df = pd.DataFrame(new_dataset_acc)
    new_df.to_parquet(args.dataset_out, engine="pyarrow")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_in", type=str)
    parser.add_argument("dataset_out", type=str)
    args = parser.parse_args()
    main(args)