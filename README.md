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