#!/bin/bash

export OMP_NUM_THREADS=4
# export CUDA_VISIBLE_DEVICES=0

# MODEL="./output/direct_T5base_allrole/20230228_155419/best_model.mdl"
MODEL="./output/direct_T5large_allrole/20230228_182631/best_model.mdl"
OUTPUT_DIR="./predictions/direct_T5large"

CONFIG=config/config_outbreak_T5-large_en.json

python ./gen/evaluate.py -c $CONFIG -m $MODEL -o $OUTPUT_DIR --beam 4
