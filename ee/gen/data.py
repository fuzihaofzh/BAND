import json, logging, pickle
import torch
from torch.utils.data import Dataset
from collections import namedtuple

logger = logging.getLogger(__name__)

gen_batch_fields = ['input_text', 'target_text', 'enc_idxs', 'enc_attn', 'enc_segs', 
                    'dec_idxs', 'dec_attn', 'lbl_idxs', 'infos']
GenBatch = namedtuple('GenBatch', field_names=gen_batch_fields, defaults=[None] * len(gen_batch_fields))

def remove_overlap_entities(entities):
    """There are a few overlapping entities in the data set. We only keep the
    first one and map others to it.
    :param entities (list): a list of entity mentions.
    :return: processed entity mentions and a table of mapped IDs.
    """
    tokens = [None] * 1000
    entities_ = []
    id_map = {}
    for entity in entities:
        start, end = entity['start'], entity['end']
        break_flag = False
        for i in range(start, end):
            if tokens[i]:
                id_map[entity['id']] = tokens[i]
                break_flag = True
        if break_flag:
            continue
        entities_.append(entity)
        for i in range(start, end):
            tokens[i] = entity['id']
    return entities_, id_map

def get_role_list(entities, events, id_map):
    entity_idxs = {entity['id']: (i,entity) for i, entity in enumerate(entities)}
    visited = [[0] * len(entities) for _ in range(len(events))]
    role_list = []
    role_list = []
    for i, event in enumerate(events):
        for arg in event['arguments']:
            entity_idx = entity_idxs[id_map.get(arg['entity_id'], arg['entity_id'])]
            
            # This will automatically remove multi role scenario
            if visited[i][entity_idx[0]] == 0:
                # ((trigger start, trigger end, trigger type), (argument start, argument end, role type))
                temp = ((event['trigger']['start'], event['trigger']['end'], event['event_type']),
                        (entity_idx[1]['start'], entity_idx[1]['end'], arg['role']))
                role_list.append(temp)
                visited[i][entity_idx[0]] = 1
    role_list.sort(key=lambda x: (x[0][0], x[1][0]))
    return role_list


class GenDataset(Dataset):
    def __init__(self, tokenizer, sep_tokens, max_length, path, max_output_length=None, unseen_types=[]):
        self.tokenizer = tokenizer
        self.sep_tokens = sep_tokens
        self.max_length = self.max_output_length = max_length
        if max_output_length is not None:
            self.max_output_length = max_output_length
        self.path = path
        self.data = []
        self.load_data(unseen_types)
        
        # these are specific to mT5
        self.in_start_code = None                      # no BOS token
        self.out_start_code = tokenizer.pad_token_id   # output starts with PAD token

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]

    def load_data(self, unseen_types):
        with open(self.path, 'rb') as f:
            data = pickle.load(f)

        for l_in, l_out, l_label in zip(data['input'], data['target'], data['all']):
            self.data.append({
                'input': l_in,
                'target': l_out,
                'all': l_label
            })
        logger.info(f'Loaded {len(self)} instances from {self.path}')

    def collate_fn(self, batch):
        input_text = [x['input'] for x in batch]
        target_text = [x['target'] for x in batch]
        
        # encoder inputs
        inputs = self.tokenizer(input_text, return_tensors='pt', padding=True, max_length=self.max_length+1, truncation=True)
        enc_idxs = inputs['input_ids']
        enc_attn = inputs['attention_mask']
        
        assert enc_idxs.size(1) < self.max_length+2

        # decoder inputs
        targets = self.tokenizer(target_text, return_tensors='pt', padding=True, max_length=self.max_output_length, truncation=True)
        dec_idxs = targets['input_ids']
        batch_size = dec_idxs.size(0)
        
        # add PAD token as the start token
        tt = torch.ones((batch_size, 1), dtype=torch.long)
        tt[:] = self.tokenizer.pad_token_id   
        dec_idxs = torch.cat((tt, dec_idxs), dim=1)
        dec_attn = torch.cat((torch.ones((batch_size, 1), dtype=torch.long), targets['attention_mask']), dim=1)
        assert dec_idxs.size(1) < self.max_output_length+2
            
        # labels
        tt = torch.ones((batch_size, 1), dtype=torch.long)
        tt[:] = self.tokenizer.pad_token_id
        raw_lbl_idxs = torch.cat((dec_idxs[:, 1:], tt), dim=1)
        lbl_attn = torch.cat((dec_attn[:, 1:], torch.zeros((batch_size, 1), dtype=torch.long)), dim=1)
        lbl_idxs = raw_lbl_idxs.masked_fill(lbl_attn==0, -100) # ignore padding
        
        # to GPU
        enc_idxs = enc_idxs.cuda()
        enc_attn = enc_attn.cuda()
        dec_idxs = dec_idxs.cuda()
        dec_attn = dec_attn.cuda()
        lbl_idxs = lbl_idxs.cuda()
        
        return GenBatch(
            input_text=input_text,
            target_text=target_text,
            enc_idxs=enc_idxs,
            enc_attn=enc_attn,
            dec_idxs=dec_idxs,
            dec_attn=dec_attn,
            lbl_idxs=lbl_idxs,
            infos=[x['all'] for x in batch]
        )
