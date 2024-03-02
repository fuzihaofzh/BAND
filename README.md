# <div align="center">BAND: Biomedical Alert News Dataset</div>
<div align="center"><b>Zihao Fu,<sup>1</sup> Meiru Zhang,<sup>1</sup> Zaiqiao Meng,<sup>2,1</sup> Yannan Shen,<sup>3</sup> David Buckeridge,<sup>3</sup> Nigel Collier<sup>1</sup></b></div>

<div align="center">
<sup>1</sup>Language Technology Lab, University of Cambridge<br>
<sup>2</sup>School of Computing Science, University of Glasgow<br>
<sup>3</sup>School of Population and Global Health, McGill University
</div>


[![PWC](https://img.shields.io/badge/%F0%9F%93%8E%20arXiv-Paper-red)](https://arxiv.org/pdf/2305.14480.pdf)
[![YouTube Video](https://img.shields.io/badge/YouTube-Video-red.svg)](https://www.youtube.com/watch?v=uvlF1RgQA-Q)
[![Slides](https://img.shields.io/badge/View-Slides-green.svg)](https://github.com/fuzihaofzh/BioNER-SynGen/blob/main/AAAI24%20-%20BAND%20Dataset%20slides.pdf)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)


## About
The Biomedical Alert News Dataset (BAND) is a well-annotated dataset aimed at improving disease surveillance and understanding of disease spread. It includes 1,508 samples from reported news articles, open emails, and alerts, along with 30 epidemiology-related questions. BAND is designed to challenge and advance NLP models in tasks like Named Entity Recognition (NER), Question Answering (QA), and Event Extraction (EE), with a focus on epidemiological analysis.

<div align="center"><img width="624" alt="image" src="https://github.com/fuzihaofzh/BAND/assets/1419566/b6b9607b-6d0c-406e-afe8-b862835c8db3"></div>


## Dataset
The BAND dataset can be fond under corresponding folders.


## Usage
### Question Answering
#### Run Decoder-Only Model
```
./scripts/train_lm.sh lmrand base
```
#### Run Encoder-Decoder Model
```
./scripts/train_seq2seq.sh band_rand ptm=t5
```

### Event Extraction
#### Run Encoder-Decoder Model
```
./scripts/train_outbreak.sh
```
#### Run Decoder-only Model
```
python train_lm.py --name "gpt2"
```

### Named Entity Recognition
#### Run Token-Based NER Model
```
./scripts/run_token_ner.sh
```
#### Run CRF-Based NER Model
```
./scripts/run_crf_ner.sh
```
#### Run Span-Based NER Model
```
./scripts/run_span_ner.sh
```

### Citation
If you find our dataset or paper useful, please cite our work:
```bibtex
@inproceedings{band2024,
  title={BAND: Biomedical Alert News Dataset},
  author={Fu, Zihao and Zhang, Meiru and Meng, Zaiqiao and Shen, Yannan and Buckeridge, David and Collier, Nigel},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2024}
}
```

## License
This project is licensed under the MIT License - refer to the [LICENSE](LICENSE) file for details.
