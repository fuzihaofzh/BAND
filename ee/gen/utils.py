import numpy as np
from tensorboardX import SummaryWriter

import os
from os.path import exists, join
import json
import torch
import re
# from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from copy import deepcopy

ROLE_LIST = ['disease', 'country', 'province', 'city', 'country Code', 'province Code', 'city Code', 'virus', 'symptoms', 'victims']
# ROLE_LIST = ['disease', 'country', 'province', 'city', 'virus', 'symptoms', 'victims']


class Recorder():
    def __init__(self, id, log=True):
        self.log = log
        now = datetime.now()
        # date = now.strftime("%y-%m-%d")
        self.dir = f"./cache_eval/{id}"
        if self.log:
            os.mkdir(self.dir)
            print("log file to ", self.dir)
            self.f = open(os.path.join(self.dir, "log.txt"), "w")
            self.writer = SummaryWriter(os.path.join(self.dir, "log"), flush_secs=60)
        
    def write_config(self, args, models, name):
        if self.log:
            with open(os.path.join(self.dir, "config.txt"), "w") as f:
                print(name, file=f)
                print(args, file=f)
                print(file=f)
                for (i, x) in enumerate(models):
                    print(x, file=f)
                    print(file=f)
        print(args)
        print()
        # for (i, x) in enumerate(models):
        #     print(x)
        #     print()

    def print(self, x=None):
        if x is not None:
            print(x)
        else:
            print()
        if self.log:
            if x is not None:
                print(x, file=self.f)
            else:
                print(file=self.f)

    def plot(self, tag, values, step):
        if self.log:
            self.writer.add_scalars(tag, values, step)


    def __del__(self):
        if self.log:
            self.f.close()
            self.writer.close()

    def save(self, model, name):
        if self.log:
            torch.save(model.state_dict(), os.path.join(self.dir, name))


class Summarizer(object):
    def __init__(self, logdir='./log'):
        self.writer = SummaryWriter(logdir)

    def scalar_summary(self, tag, value, step):
        self.writer.add_scalar(tag, value, step)

    def text_summary(self, tag, value, step):
        self.writer.add_text(tag, value, step)


def generate_vocabs(datasets):
    event_type_set = set()
    role_type_set = set()
    for dataset in datasets:
        event_type_set.update(dataset.event_type_set)
        role_type_set.update(dataset.role_type_set)
    
    event_type_itos = sorted(event_type_set)
    role_type_itos = sorted(role_type_set)
    
    event_type_stoi = {k: i for i, k in enumerate(event_type_itos)}
    role_type_stoi = {k: i for i, k in enumerate(role_type_itos)}
    
    return {
        'event_type_itos': event_type_itos,
        'event_type_stoi': event_type_stoi,
        'role_type_itos': role_type_itos,
        'role_type_stoi': role_type_stoi,
    }


def safe_div(num, denom):
    if denom > 0:
        return num / denom
    else:
        return 0


def compute_f1(predicted, gold, matched):
    precision = safe_div(matched, predicted)
    recall = safe_div(matched, gold)
    f1 = safe_div(2 * precision * recall, precision + recall)
    return precision, recall, f1


def cal_scores(gold_roles, pred_roles):
    # argument identification
    gold_arg_id_num, pred_arg_id_num, match_arg_id_num = 0, 0, 0
    for gold_role, pred_role in zip(gold_roles, pred_roles):
        gold_set = set([(r[0][2],)+r[1][:-1] for r in gold_role])
        pred_set = set([(r[0][2],)+r[1][:-1] for r in pred_role])
        
        gold_arg_id_num += len(gold_set)
        pred_arg_id_num += len(pred_set)
        match_arg_id_num += len(gold_set & pred_set)
        
    # argument classification
    gold_arg_cls_num, pred_arg_cls_num, match_arg_cls_num = 0, 0, 0
    for gold_role, pred_role in zip(gold_roles, pred_roles):
        gold_set = set([(r[0][2],)+r[1] for r in gold_role])
        pred_set = set([(r[0][2],)+r[1] for r in pred_role])
        
        gold_arg_cls_num += len(gold_set)
        pred_arg_cls_num += len(pred_set)
        match_arg_cls_num += len(gold_set & pred_set)
    
    scores = {
        'arg_id': (gold_arg_id_num, pred_arg_id_num, match_arg_id_num) + compute_f1(pred_arg_id_num, gold_arg_id_num, match_arg_id_num),
        'arg_cls': (gold_arg_cls_num, pred_arg_cls_num, match_arg_cls_num) + compute_f1(pred_arg_cls_num, gold_arg_cls_num, match_arg_cls_num),
    }
    
    return scores


def cal_all_f1(pred_attrs, gold_attrs):
    # pred_labels = {c: [] for c in ROLE_LIST}
    # gold_labels = {c: [] for c in ROLE_LIST}
    # match_attribute = {c: [] for c in ROLE_LIST}
    overall_f1 = {c: [] for c in ROLE_LIST}

    # for p in pred_attrs:
    #     if p[1] in ROLE_LIST:
    #         pred_labels[p[1]].append(p[0])
    # for g in gold_attrs:
    #     if g[1] in ROLE_LIST:
    #         gold_labels[g[1]].append(g[0])

    # for r in ROLE_LIST:
    #     if pred_labels[r] != []:
    #         gold_label_copy = deepcopy(gold_labels[r])
    #         for p_attr in pred_labels[r]:
    #             if p_attr in gold_label_copy:
    #                 match_attribute[r].append(p_attr)
    #                 gold_label_copy.remove(p_attr)
        

    return overall_f1


def extract_outbreak(text):
    output = []
    tag_s = re.search('<\|[^/>][^>]*\|>', text)
    while tag_s:
        text = text[tag_s.end():]
        r_type = tag_s.group()[2:-2]
        if r_type in ROLE_LIST:
            tag_e = re.search(f'<\|/{r_type}\|>', text)
            if tag_e:
                arg = text[:tag_e.start()].strip()
                for a in arg.split(f' [And] '):
                    a = a.strip()
                    if a != '' and a != "[None]":
                        output.append((a.lower(), r_type))
                text = text[tag_e.end():]
        tag_s = re.search('<\|[^/>][^>]*\|>', text)
    return output


def get_span_idxs(pieces, token_start_idxs, span, tokenizer, trigger_span=None):
    """
    Convert the predicted span to the offsets in the passage.
    Return (-1, -1) if there is no matched string.
    """
    
    t_span = span.split(' ')
    
    words = []
    for s in t_span:
        words.extend(tokenizer.encode(s, add_special_tokens=True)[:-1]) # ignore eos
    
    candidates = []
    for i in range(len(pieces)):
        j = 0
        k = 0
        while j < len(words) and i+k < len(pieces):
            if pieces[i+k] == words[j]:
                j += 1
                k += 1
            elif tokenizer.decode(words[j]) == "":
                j += 1
            elif tokenizer.decode(pieces[i+k]) == "":
                k += 1
            else:
                break
        if j == len(words):
            candidates.append((i, i+k))
            
    candidates = [(token_start_idxs.index(c1), token_start_idxs.index(c2))for c1, c2 in candidates if c1 in token_start_idxs and c2 in token_start_idxs]
    if len(candidates) < 1:
        return -1, -1
    else:
        if trigger_span is None:
            return candidates[0]
        else:
            return sorted(candidates, key=lambda x: np.abs(trigger_span[0]-x[0]))[0]
        
def get_span_idxs_zh(tokens, span, trigger_span=None):
    """
    Convert the predicted span to the offsets in the passage for Chinese.
    Return (-1, -1) if there is no matched string.
    """
    
    candidates = []
    for i in range(len(tokens)):
        for j in range(i, len(tokens)):
            c_string = "".join(tokens[i:j+1])
            if c_string == span:
                candidates.append((i, j+1))
                break
            elif not span.startswith(c_string):
                break
                
    if len(candidates) < 1:
        return -1, -1
    else:
        if trigger_span is None:
            return candidates[0]
        else:
            return sorted(candidates, key=lambda x: np.abs(trigger_span[0]-x[0]))[0]