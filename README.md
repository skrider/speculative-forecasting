# Speculative Forecasting

This repo contains some experiments for controlling how many tokens to predict ahead for speculative decoding. The RL environment scaffolding and DQN implementation is from CS 285 at UC Berkeley.

## Install

```
conda create --name dman
conda activate dman
conda install python=3.10 swig
pip install -r requirements.txt
pip install -e .
```

## Generate Dataset

This repo makes heavy use of offline preprocessing. For now, we preprocess all sequences with `scripts/process_dataset.py` and cache the main and draft model hidden states for each token. To generate the dataset, download [lmsys chat 1m](https://huggingface.co/datasets/lmsys/lmsys-chat-1m) and run `scripts/process_dataset.py` to generate the caches.

