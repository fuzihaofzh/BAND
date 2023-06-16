# BAND
A well-annotated Biomedical Alert News Dataset

## Question Answering
### Run Decoder-Only Model
```
./scripts/train_lm.sh lmrand base
```

### Run Encoder-Decoder Model
```
./scripts/train_seq2seq.sh band_rand ptm=t5
```


## Event Extraction
### Run Encoder-Decoder Model
```
./scripts/train_outbreak.sh
```

### Run Decoder-only Model
```
python train_lm.py --name "gpt2"
```

## Named Entity Recognition

### Run Token-Based NER Model

```
./scripts/run_token_ner.sh
```

### Run CRF-Based NER Model

```
./scripts/run_crf_ner.sh
```

### Run Span-Based NER Model

```
./scripts/run_span_ner.sh
```