#!/bin/bash


#DATA=lmrand
#DATA=lmstratified
DATA=$1
MODE=$2
EXP="$DATA"__"$MODE"
DATA_DIR=data/$DATA
OUTPUT_DIR=output/$EXP

#facebook/opt-125m
#decapoda-research/llama-7b-hf
#gpt2

if [[ -z "$CUDA_VISIBLE_DEVICES" ]]; then
    export CUDA_VISIBLE_DEVICES=$(get_free_gpu.sh)
fi

echo "CUDA_VISIBLE_DEVICES:" $CUDA_VISIBLE_DEVICES

if [ "$DBG" = 1 ]; then
    export DBG=' -m ptvsd --host 0.0.0.0 --port 5678 --wait '
	EXP=dbg
	BSZ=20
else
    BSZ=1250 #230
fi

mkdir -p $OUTPUT_DIR
echo $OUTPUT_DIR

# facebook/galactica-125m
# EleutherAI/gpt-neo-125m
# bigscience/bloom-560m

python $DBG src/run_clm.py --model_name_or_path gpt2 --train_file $DATA_DIR/train.txt --validation_file $DATA_DIR/dev.txt --per_device_train_batch_size 1 --per_device_eval_batch_size 5 --do_train --do_eval --output_dir $OUTPUT_DIR --overwrite_output_dir --save_steps 100000000 --predict_file $DATA_DIR/test.txt --num_train_epochs 3 --fp16 --usermode $MODE

# Predict
python $DBG src/run_clm.py --model_name_or_path $OUTPUT_DIR --train_file $DATA_DIR/train.txt --validation_file $DATA_DIR/dev.txt --predict_file $DATA_DIR/test.txt --per_device_train_batch_size 5 --per_device_eval_batch_size 5 --do_predict --output_dir $OUTPUT_DIR --overwrite_output_dir --save_steps 1000000 --fp16 --usermode $MODE

python src/eval_lm_score.py $DATA $MODE
echo $EXP
