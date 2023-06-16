#!/bin/bash


BERT_BASE_DIR=bert-base-cased
DATA=bandcrfstratified
DATA_DIR=data/$DATA
OUTPUT_DIR=output/"$DATA"


mkdir -p $OUTPUT_DIR
echo $OUTPUT_DIR

python -u $DBG BERT-BiLSTM-CRF-NER-pytorch/ner.py --model_name_or_path ${BERT_BASE_DIR} --do_train True --do_eval True --do_test True --max_seq_length 256 --train_file ${DATA_DIR}/train.txt --eval_file ${DATA_DIR}/dev.txt --test_file ${DATA_DIR}/test.txt --train_batch_size 32 --eval_batch_size 32 --num_train_epochs 10 --do_lower_case --logging_steps 200 --need_birnn True --rnn_dim 256 --clean True --output_dir $OUTPUT_DIR 2>&1 | tee output/log.txt

#python -u $DBG BERT-BiLSTM-CRF-NER-pytorch/ner.py --model_name_or_path ${BERT_BASE_DIR} --do_test True --max_seq_length 256 --test_file ${DATA_DIR}/test.txt --train_batch_size 32 --eval_batch_size 32 --num_train_epochs 3 --do_lower_case --logging_steps 200 --need_birnn True --rnn_dim 256 --clean True --output_dir $OUTPUT_DIR

mv output/log.txt $OUTPUT_DIR/log.txt