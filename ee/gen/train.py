import os, sys, json, logging, time, pprint, tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import MT5Tokenizer, AdamW, get_linear_schedule_with_warmup, T5Tokenizer
from transformers import BartTokenizer
from model import GenerativeModel
from data import GenDataset
from utils import Summarizer, compute_f1, extract_outbreak, cal_all_f1
from argparse import ArgumentParser, Namespace
import re
from copy import deepcopy

# configuration
parser = ArgumentParser()
parser.add_argument('-c', '--config', required=True)
args = parser.parse_args()
with open(args.config) as fp:
    config = json.load(fp)
config.update(args.__dict__)
config = Namespace(**config)

# fix random seed
np.random.seed(config.seed)
torch.manual_seed(config.seed)
torch.backends.cudnn.enabled = False

# set GPU device
if isinstance(config.gpu_device, int):
    torch.cuda.set_device(config.gpu_device)

# logger and summarizer
timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
output_dir = os.path.join(config.output_dir, timestamp)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
log_path = os.path.join(output_dir, "train.log")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(message)s', datefmt='[%Y-%m-%d %H:%M:%S]', 
                    handlers=[logging.FileHandler(os.path.join(output_dir, "train.log")), logging.StreamHandler()])
logger = logging.getLogger(__name__)
logger.info(f"\n{pprint.pformat(vars(config), indent=4)}")
summarizer = Summarizer(output_dir)

# output
with open(os.path.join(output_dir, 'config.json'), 'w') as fp:
    json.dump(vars(config), fp, indent=4)
best_model_path = os.path.join(output_dir, 'best_model.mdl')
dev_prediction_path = os.path.join(output_dir, 'pred.dev.json')
test_prediction_path = os.path.join(output_dir, 'pred.test.json')

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
elif config.model_name.startswith("facebook/bart-"):
    # Load the BART tokenizer and model
    tokenizer = BartTokenizer.from_pretrained(config.model_name, cache_dir=config.cache_dir)
else:
    raise NotImplementedError

special_tokens = ['<|disease|>', '<|/disease|>', '<|country|>', '<|/country|>', '<|province|>', '<|/province|>', '<|city|>', '<|/city|>', 
            '<|country Code|>', '<|/country Code|>', '<|province Code|>', '<|/province Code|>', '<|city Code|>', '<|/city Code|>', '<|virus|>', 
            '<|/virus|>', '<|symptoms|>', '<|/symptoms|>', '<|victims|>', '<|/victims|>', '[None]', '[And]']
# special_tokens = ['<|disease|>', '<|/disease|>', '<|country|>', '<|/country|>', '<|province|>', '<|/province|>', '<|city|>', '<|/city|>', 
#             '<|virus|>', '<|/virus|>', '<|symptoms|>', '<|/symptoms|>', '<|victims|>', '<|/victims|>', '[None]', '[And]']

sep_tokens = []

tokenizer.add_tokens(sep_tokens+special_tokens)
    
# load data
train_set = GenDataset(tokenizer, sep_tokens, config.max_length, config.train_finetune_file, config.max_output_length)
dev_set = GenDataset(tokenizer, sep_tokens, config.max_length, config.dev_finetune_file, config.max_output_length)
test_set = GenDataset(tokenizer, sep_tokens, config.max_length, config.test_finetune_file, config.max_output_length)
train_batch_num = len(train_set) // config.train_batch_size + (len(train_set) % config.train_batch_size != 0)
dev_batch_num = len(dev_set) // config.eval_batch_size + (len(dev_set) % config.eval_batch_size != 0)
test_batch_num = len(test_set) // config.eval_batch_size + (len(test_set) % config.eval_batch_size != 0)

# initialize the model
model = GenerativeModel(config, tokenizer)
if isinstance(config.gpu_device, int):
    model.cuda(device=config.gpu_device)
else:
    if len(config.gpu_device) > 1:
        os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_device
    model = torch.nn.DataParallel(model)
    model.cuda()

# optimizer
param_groups = [{'params': model.parameters(), 'lr': config.learning_rate, 'weight_decay': config.weight_decay}]
optimizer = AdamW(params=param_groups)
schedule = get_linear_schedule_with_warmup(optimizer,
                                           num_warmup_steps=train_batch_num*config.warmup_epoch,
                                           num_training_steps=train_batch_num*config.max_epoch)


# start training
logger.info("Start training ...")
summarizer_step = 0
best_dev_epoch = -1
best_dev_scores = {
    'overall_f1': (0.0, 0.0, 0.0),
}
for epoch in range(1, config.max_epoch+1):
    logger.info(log_path)
    logger.info(f"Epoch {epoch}")
    
    # training
    progress = tqdm.tqdm(total=train_batch_num, ncols=75, desc='Train {}'.format(epoch))
    model.train()
    optimizer.zero_grad()
    for batch_idx, batch in enumerate(DataLoader(train_set, batch_size=config.train_batch_size // config.accumulate_step,
                                                 shuffle=True, drop_last=False, collate_fn=train_set.collate_fn)):
        
        # forard model
        #print(batch) # GenBatch(input_text=[], enc_idxs=[], enc_segs=None, dec_idxs=[], dec_attn=[], lbl_idxs=[])
        loss = model(batch)

        # record loss
        summarizer.scalar_summary('train/loss', loss.mean(), summarizer_step)
        summarizer_step += 1
        
        # TODO: Contrastive Trigger beam search
        loss = loss * (1 / config.accumulate_step)
        loss.backward(torch.ones_like(loss))

        if (batch_idx + 1) % config.accumulate_step == 0:
            progress.update(1)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clipping)
            optimizer.step()
            schedule.step()
            optimizer.zero_grad()
            
    progress.close()

    # eval dev set
    progress = tqdm.tqdm(total=dev_batch_num, ncols=75, desc='Dev {}'.format(epoch))
    model.eval()
    best_dev_flag = False
    write_output = []
    dev_gold_attribute = [] # [[(value, role)
    dev_pred_attribute = [] # (value, role)
    dev_match_attribute = [] 
    
    for batch_idx, batch in enumerate(DataLoader(dev_set, batch_size=config.eval_batch_size, 
                                                 shuffle=False, collate_fn=dev_set.collate_fn)):
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

            write_output.append({
                'input text': i_text, 
                'gold text': g_text,
                'pred text': p_text,
                'gold attribute': gold_attr,
                'pred attribute': pred_attr,
                'original label': label_ori
            })

    progress.close()

    dev_all_f1 = cal_all_f1(dev_pred_attribute, dev_gold_attribute) # a dictionary {role_type: (P, R, F1)}
    dev_scores = {
        'overall_f1': compute_f1(len(dev_pred_attribute), len(dev_gold_attribute), len(dev_match_attribute)),
        'attribute_f1': dev_all_f1
    }

    # print scores
    print("||||||||||||||||||||||||||||||||||-")
    print('Overall F1    - P: {:5.2f} ({:4d}/{:4d}), R: {:5.2f} ({:4d}/{:4d}), F: {:5.2f}'.format(
        dev_scores['overall_f1'][0] * 100.0, len(dev_match_attribute), len(dev_pred_attribute), 
        dev_scores['overall_f1'][1] * 100.0, len(dev_match_attribute), len(dev_gold_attribute), dev_scores['overall_f1'][2] * 100.0))
    print("||||||||||||||||||||||||||||||||||-")
    print("F1 of all attributes")
    print(dev_all_f1)
    print("||||||||||||||||||||||||||||||||||-")
    
    # check best dev model
    if dev_scores['overall_f1'][2] >= best_dev_scores['overall_f1'][2]:
        best_dev_flag = True
        
    # if best dev, save model and evaluate test set
    if best_dev_flag:    
        best_dev_scores = dev_scores
        best_dev_epoch = epoch
        
        # save best model
        logger.info('Saving best model')
        if not isinstance(config.gpu_device, int) and len(config.gpu_device)>1:
            torch.save(model.module.state_dict(), best_model_path)
        else:
            torch.save(model.state_dict(), best_model_path)

        # save dev result
        with open(dev_prediction_path, 'w') as fp:
            json.dump(write_output, fp, indent=4)

        # eval test set
        progress = tqdm.tqdm(total=test_batch_num, ncols=75, desc='Test {}'.format(epoch))
        write_output = []
        test_gold_attribute = [] # (gold_attr, role)
        test_pred_attribute = [] # (pred_attr, role))
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

                write_output.append({
                    'input text': i_text, 
                    'gold text': g_text,
                    'pred text': p_text,
                    'gold attribute': gold_attr,
                    'pred attribute': pred_attr,
                    'original label': label_ori
                })
        progress.close()

        test_all_f1 = cal_all_f1(test_pred_attribute, test_gold_attribute)
        test_scores = {
            'overall_f1': compute_f1(len(test_pred_attribute), len(test_gold_attribute), len(test_match_attribute)),
            'attribute_f1': test_all_f1
        }

        # print scores
        print("||||||||||||||||||||||||||||||||||-")
        print('Triple F1    - P: {:5.2f} ({:4d}/{:4d}), R: {:5.2f} ({:4d}/{:4d}), F: {:5.2f}'.format(
            test_scores['overall_f1'][0] * 100.0, len(test_match_attribute), len(test_pred_attribute), 
            test_scores['overall_f1'][1] * 100.0, len(test_match_attribute), len(test_gold_attribute), test_scores['overall_f1'][2] * 100.0))
        print("||||||||||||||||||||||||||||||||||-")
        print("F1 of all attributes")
        print(test_all_f1)
        print("||||||||||||||||||||||||||||||||||-")
        
        # save test result
        with open(test_prediction_path, 'w') as fp:
            json.dump(write_output, fp, indent=4)
            
    logger.info({"epoch": epoch, "dev_scores": dev_scores})
    if best_dev_flag:
        logger.info({"epoch": epoch, "test_scores": test_scores})
    logger.info("Current best")
    logger.info({"best_epoch": best_dev_epoch, "best_scores": best_dev_scores})
        
logger.info(log_path)
logger.info("Done!")
