#!/bin/bash
DATASET=$1 #squad #band_rand
EXP=$1__$2

if [ "$DBG" = 1 ]; then
    export DBG=' -m ptvsd --host 0.0.0.0 --port 5678 --wait '
else
    export NUMWORDER=5
fi

if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    export CUDA_VISIBLE_DEVICES=$(get_free_gpu.sh)
fi

DATASET_PATH=data/$DATASET
if [[ "${DATASET}" == "squad" ]]; then
    DATASET_PATH=squad_v2
    EXP=squad_v2
fi

echo output/$EXP

python $DBG src/run_seq2seq_qa.py --model_name_or_path output/$EXP --dataset_name $DATASET_PATH --context_column context --question_column question --per_device_eval_batch_size 1 --answer_column answers --do_predict --predict_with_generate --version_2_with_negative --learning_rate 3e-5 --num_train_epochs 6 --max_seq_length 384 --doc_stride 128 --output_dir output/$EXP