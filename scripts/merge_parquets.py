import pandas as pd 
import sys
import os

def main(dst, srcs):
    for src in srcs:
        df = pd.read_parquet(src)
        if 'fragment' not in df:
            df['fragment'] = src
        # append
        if os.path.exists(dst):
            df.to_parquet(dst, engine='fastparquet', append=True)
        else:
            df.to_parquet(dst, engine='fastparquet')

if __name__ == "__main__":
    dst = sys.argv[1]
    srcs = sys.argv[2:]

    main(dst, srcs)