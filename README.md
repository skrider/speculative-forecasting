# Draftsman

Download dataset:

```
wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json
```

TODO export dataset to parquet

## install

```
conda create --name dman
conda activate dman
conda install python=3.10 swig
pip install -r requirements.txt
pip install -e .
```

Get dataset:

```
bash scripts/download_chatbot_arena.sh
python process_chatbot_arena.py /tmp/dataset.json datasets/chatbot_arena_2.parquet
```


## Run single node, multi GPU

```
# start the ray head node for running main model on vLLM
conda activate dman
CUDA_VISIBLE_DEVICES=4,5,6,7 ray start --head --num-gpus=4

# export huggingface token
export HUGGING_FACE_HUB_TOKEN=...

# run experiment, ensuring no GPUs conflict
python draftsman/scripts/run_draftsman.py -cfg experiments/spec/spec_rand_honeydew.yaml --which_gpu 3 --dataset_dir datasets
```

This will output nothing for a while because it is running an eval rollout which takes up to a minute. To avoid this append `-neval 0`

## Run multi node, multi GPU

```
# on the head node, which is assumed to have a GPU:
conda activate dman
# prevent scheduling of ray functions onto the head node, reserve its GPU
# for non-ray-managed stuff
ray start --head --num-gpus 0

# on the worker nodes, which are assumed to have GPUs
conda activate dman
# use address from output of last command
ray start --address ADDRESS

export HUGGING_FACE_HUB_TOKEN=...

# on the head node
python draftsman/scripts/run_draftsman.py -cfg experiments/spec/spec_rand.yaml --dataset_dir datasets
```

## Generate graphs

Create an environment with tensorboard and matplotlib if you don't have one yet:

```
conda create --name dman-graph
conda activate dman-graph
conda install --solver libmamba python=3.10
pip install tensorboard tensorflow matplotlib
```
