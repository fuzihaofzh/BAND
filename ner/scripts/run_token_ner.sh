#!/bin/bash

DATA=bandtokenstratified
EXP=$DATA

OUTPUT_DIR=output/$EXP
mkdir -p $OUTPUT_DIR

python3 src/tf_ner.py --model_name_or_path bert-base-cased --output_dir $OUTPUT_DIR --do_train --do_eval --do_predict --overwrite_output_dir --save_steps 10000000000 --train_file data/$DATA/train.json --validation_file data/$DATA/dev.json --test_file data/$DATA/test.json 2>&1 | tee $OUTPUT_DIR/log.txt 


#python3 src/tf_ner.py --model_name_or_path bert-base-uncased --dataset_name conll2003 --output_dir output/tf_ner --do_train --do_eval --overwrite_output_dir --save_steps 10000000000