#!/bin/bash

export TRAIN_FILE="./corpus.txt"

python finetune_bert.py \
    --output_dir=output \
    --model_type=bert \
    --model_name_or_path=bert-base-uncased \
    --do_train \
    --train_data_file="./corpus.txt" \
    --mlm
