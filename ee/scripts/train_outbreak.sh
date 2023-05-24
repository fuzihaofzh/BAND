#!/bin/bash

export OMP_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=1

# CONFIG=config/config_outbreak_T5-base_en.json
# CONFIG=config/config_outbreak_T5-large_en.json
# CONFIG=config/config_outbreak_T5-small_en.json
# CONFIG=config/config_outbreak_T5-large_en_stratified.json
CONFIG=config/config_outbreak_bart-base_en.json

python ./gen/train.py -c $CONFIG

