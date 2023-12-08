import time
import argparse
import pickle
import ray

from draftsman.agents import agents as agent_types

import os
import time

import gym
import numpy as np
import torch
from draftsman.infrastructure import pytorch_util as ptu
import tqdm

def export_dataset(args: argparse.Namespace):
    # TODO figure out how I saved the dataset to parquet
    pass

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)

    args = parser.parse_args()

    export_dataset(args)

if __name__ == "__main__":
    main()
