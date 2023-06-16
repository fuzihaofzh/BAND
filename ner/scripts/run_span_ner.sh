#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2    # There are 32 CPU cores on GPU nodes
#SBATCH --mem=50000               # Request the full memory of the node
#SBATCH --time=3:00
#SBATCH --partition=infofil01


DATA=bandspanstratified
EXP=$DATA


echo CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES
mkdir -p output/"$EXP"

python -u $DBG pure/run_entity.py \
    --do_eval --eval_test --do_train \
    --context_window 0 \
    --task band \
    --data_dir data/$DATA \
    --output_dir output/"$EXP" \
    --model bert-base-cased \
    --num_epoch 30 \
    --train_batch_size 40 2>&1 | tee output/"$EXP"/log.txt

#--model dmis-lab/biobert-base-cased-v1.2
#--model allenai/scibert_scivocab_uncased \