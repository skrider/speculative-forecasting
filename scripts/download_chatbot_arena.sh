#!/bin/bash

URL="https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json"
DOWNLOAD=/tmp/dataset.json

wget $URL --output-document=- | jq .[:20000] > $DOWNLOAD
