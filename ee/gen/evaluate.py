import os, sys, json, logging, pprint, tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import MT5Tokenizer, T5Tokenizer
from model import GenerativeModel, Prefix_fn_cls
from data import GenDataset
from utils import cal_scores, get_span_idxs, get_span_idxs_zh, compute_f1, extract_outbreak
from argparse import ArgumentParser, Namespace
from copy import deepcopy

# configuration
parser = ArgumentParser()
parser.add_argument('-c', '--config', required=True)
parser.add_argument('-m', '--model', required=True)
parser.add_argument('-o', '--output_dir', type=str, required=True)
parser.add_argument('--constrained_decode', default=False, action='store_true')
parser.add_argument('--beam', type=int, default=4)
args = parser.parse_args()
with open(args.config) as fp:
    config = json.load(fp)
config = Namespace(**config)

# over write beam size
config.beam_size = args.beam

# fix random seed
np.random.seed(config.seed)
torch.manual_seed(config.seed)
torch.backends.cudnn.enabled = False

# set GPU device
# set GPU device
if isinstance(config.gpu_device, int):
    torch.cuda.set_device(config.gpu_device)

# logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(message)s', datefmt='[%Y-%m-%d %H:%M:%S]')
logger = logging.getLogger(__name__)

# tokenizer
if config.model_name.startswith("google/mt5-"):
    tokenizer = MT5Tokenizer.from_pretrained(config.model_name, cache_dir=config.cache_dir)
elif config.model_name.startswith("copy+google/mt5-"):
    model_name = config.model_name.split('copy+', 1)[1]
    tokenizer = MT5Tokenizer.from_pretrained(model_name, cache_dir=config.cache_dir)
elif config.model_name.startswith("t5-"):
    tokenizer = T5Tokenizer.from_pretrained(config.model_name, cache_dir=config.cache_dir)
elif config.model_name.startswith("copy+t5-"):
    model_name = config.model_name.split('copy+', 1)[1]
    tokenizer = T5Tokenizer.from_pretrained(model_name, cache_dir=config.cache_dir)
else:
    raise NotImplementedError



# special_tokens = ['<|disease|>', '<|/disease|>', '<|country|>', '<|/country|>', '<|province|>', '<|/province|>', '<|city|>', '<|/city|>', 
#             '<|country Code|>', '<|/country Code|>', '<|province Code|>', '<|/province Code|>', '<|city Code|>', '<|/city Code|>', '<|virus|>', 
#             '<|/virus|>', '<|symptoms|>', '<|/symptoms|>', '<|victims|>', '<|/victims|>', '[None]', '[And]']
special_tokens = ['<|disease|>', '<|/disease|>', '<|country|>', '<|/country|>', '<|province|>', '<|/province|>', '<|city|>', '<|/city|>', 
            '<|virus|>', '<|/virus|>', '<|symptoms|>', '<|/symptoms|>', '<|victims|>', '<|/victims|>', '[None]', '[And]']
sep_tokens = []

tokenizer.add_tokens(sep_tokens+special_tokens)


# load data
dev_set = GenDataset(tokenizer, [], config.max_length, config.dev_finetune_file, config.max_output_length)
test_set = GenDataset(tokenizer, [], config.max_length, config.test_finetune_file, config.max_output_length)
dev_batch_num = len(dev_set) // config.eval_batch_size + (len(dev_set) % config.eval_batch_size != 0)
test_batch_num = len(test_set) // config.eval_batch_size + (len(test_set) % config.eval_batch_size != 0)

# load model
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"
logger.info(f"Loading model from {args.model}")
model = GenerativeModel(config, tokenizer)
# model.load_state_dict(torch.load(args.model, map_location=f'cuda:{config.gpu_device}'))
model.load_state_dict(torch.load(args.model))
model.cuda(device=config.gpu_device)
model.eval()

# output directory
if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

# eval dev set
progress = tqdm.tqdm(total=dev_batch_num, ncols=75, desc='Dev')
dev_output = []
dev_gold_attribute = [] # [[(value, role)
dev_pred_attribute = [] # (value, role)
dev_match_attribute = [] 

count = 0
for batch in DataLoader(dev_set, batch_size=config.eval_batch_size, shuffle=False, collate_fn=dev_set.collate_fn):
    progress.update(1)
    if not isinstance(config.gpu_device, int) and len(config.gpu_device)>1:
        pred_text = model.module.predict(batch, num_beams=config.beam_size, max_length=config.max_output_length)
    else:
        pred_text = model.predict(batch, num_beams=config.beam_size, max_length=config.max_output_length)
            
    gold_text = batch.target_text
    input_text = batch.input_text
    

    for i_text, g_text, p_text, label_ori in zip(input_text, gold_text, pred_text, batch.infos):
        gold_attr = extract_outbreak(g_text)
        # this is attribute_all, entity pair level f1 score
        dev_gold_attribute.extend(gold_attr)
        pred_attr = extract_outbreak(p_text)

        if pred_attr != []:
            dev_pred_attribute.extend(pred_attr)
            gold_attr_copy = deepcopy(gold_attr)
            for p_attr in pred_attr:
                if p_attr in gold_attr_copy:
                    dev_match_attribute.append(p_attr)
                    gold_attr_copy.remove(p_attr)

        dev_output.append({
            'input text': i_text, 
            'gold text': g_text,
            'pred text': p_text,
            'gold attribute': gold_attr,
            'pred attribute': pred_attr,
            'original label': label_ori
        })

progress.close()

# calculate scores
print(len(dev_pred_attribute), len(dev_gold_attribute), len(dev_match_attribute))
dev_scores = {
    'overall_f1': compute_f1(len(dev_pred_attribute), len(dev_gold_attribute), len(dev_match_attribute)),
}


# print scores
print("||||||||||||||||||||||||||||||||||-")
print('Overall F1    - P: {:5.2f} ({:4d}/{:4d}), R: {:5.2f} ({:4d}/{:4d}), F: {:5.2f}'.format(
    dev_scores['overall_f1'][0] * 100.0, len(dev_match_attribute), len(dev_pred_attribute), 
    dev_scores['overall_f1'][1] * 100.0, len(dev_match_attribute), len(dev_gold_attribute), dev_scores['overall_f1'][2] * 100.0))
print("||||||||||||||||||||||||||||||||||-")
    

with open(os.path.join(args.output_dir, 'dev.pred.json'), 'w') as fp:
    json.dump(dev_output, fp, indent=2)
    
    
# test set
progress = tqdm.tqdm(total=test_batch_num, ncols=75, desc='Test')
test_output = []
test_gold_attribute = [] # (gold_ent1, gold_ent2, gold_sentiment)
test_pred_attribute = [] # (pred_ent1, pred_ent2, pred_sentiment)
test_match_attribute = [] 

for batch_idx, batch in enumerate(DataLoader(test_set, batch_size=config.eval_batch_size, 
                                            shuffle=False, collate_fn=test_set.collate_fn)):
    progress.update(1)
    if not isinstance(config.gpu_device, int) and len(config.gpu_device)>1:
        pred_text = model.module.predict(batch, num_beams=config.beam_size, max_length=config.max_output_length)
    else:
        pred_text = model.predict(batch, num_beams=config.beam_size, max_length=config.max_output_length)
    
    gold_text = batch.target_text
    input_text = batch.input_text
    

    for i_text, g_text, p_text, label_ori in zip(input_text, gold_text, pred_text, batch.infos):
        # get gold attribute
        gold_attr = extract_outbreak(g_text)
        # this is attribute_all, entity pair level f1 score
        test_gold_attribute.extend(gold_attr)
        pred_attr = extract_outbreak(p_text)

        # for overall f1
        if pred_attr != []:
            test_pred_attribute.extend(pred_attr)
            gold_attr_copy = deepcopy(gold_attr)
            for p_attr in pred_attr:
                if p_attr in gold_attr_copy:
                    gold_attr_copy.remove(p_attr)
                    test_match_attribute.append(p_attr)

        test_output.append({
            'input text': i_text, 
            'gold text': g_text,
            'pred text': p_text,
            'gold attribute': gold_attr,
            'pred attribute': pred_attr,
            'original label': label_ori
        })

progress.close()

print(len(test_pred_attribute), len(test_gold_attribute), len(test_match_attribute))
test_scores = {
    'overall_f1': compute_f1(len(test_pred_attribute), len(test_gold_attribute), len(test_match_attribute)),
}

# print scores
print("||||||||||||||||||||||||||||||||||-")
print('Triple F1    - P: {:5.2f} ({:4d}/{:4d}), R: {:5.2f} ({:4d}/{:4d}), F: {:5.2f}'.format(
    test_scores['overall_f1'][0] * 100.0, len(test_match_attribute), len(test_pred_attribute), 
    test_scores['overall_f1'][1] * 100.0, len(test_match_attribute), len(test_gold_attribute), test_scores['overall_f1'][2] * 100.0))
print("||||||||||||||||||||||||||||||||||-")


# write outputs
with open(os.path.join(args.output_dir, 'test.pred.json'), 'w') as fp:
    json.dump(test_output, fp, indent=2)
