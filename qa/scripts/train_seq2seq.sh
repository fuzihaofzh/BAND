#!/bin/bash
DATASET=$1 #band_rand
EXP="$DATASET"__$2

BSZ=2

if [ "$DBG" = 1 ]; then
    export DBG=' -m ptvsd --host 0.0.0.0 --port 5678 --wait '
    BSZ=2
else
    export NUMWORDER=5
fi

if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    export CUDA_VISIBLE_DEVICES=$(get_free_gpu.sh)
fi
echo CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES
DATASET_PATH=data/$DATASET

if [[ "${DATASET}" == "squad" ]]; then
    DATASET_PATH=squad_v2
    EXP=squad_v2__$2
fi

echo output/$EXP


# t5-base
# google/flan-t5-base

python $DBG src/run_seq2seq_qa.py --model_name_or_path google/flan-t5-base --dataset_name $DATASET_PATH --context_column context --question_column question --answer_column answers  --do_predict --per_device_train_batch_size $BSZ --learning_rate 3e-5 --num_train_epochs 1 --max_seq_length 1024 --doc_stride 128 --save_steps 100000000 --output_dir output/$EXP --overwrite_output_dir --predict_with_generate --version_2_with_negative --save_total_limit 1 --fp16 --do_train --do_eval --usermode $2


echo output/$EXP