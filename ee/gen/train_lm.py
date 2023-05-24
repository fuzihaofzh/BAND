import json
import torch
import pickle
from torch.utils.data import Dataset, DataLoader
from transformers import (
    GPT2Config,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    AutoModelForCausalLM, 
    AutoTokenizer,
    AutoConfig,
)
import numpy as np
import random
import os
import argparse

os.environ['CUDA_VISIBLE_DEVICES']="0"

## dataset loader
class EventExtractionDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=256):
        ## tokenize and store the data

        self.tokenizer = tokenizer
        self.max_length = max_length

        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        
        self.input_ids = []
        self.attn_masks = []
        
        for l_in, l_out, l_label in zip(data['input'], data['target'], data['all']):
            # prepare the text
            prep_txt = f'<startoftext> Document: {l_in} <sep> Event: {l_out} <endoftext>'
            # tokenize
            encodings_dict = self.tokenizer(prep_txt, truncation=True, max_length=self.max_length, padding='max_length')

            # append to list
            self.input_ids.append(torch.tensor(encodings_dict['input_ids']))
            self.attn_masks.append(torch.tensor(encodings_dict['attention_mask']))


    def __len__(self):
        ## return the length of the total dataset
        return len(self.input_ids)

    def __getitem__(self, idx):
        ## fetch one data and then return it
        return self.input_ids[idx], self.attn_masks[idx]

def seed_everything(seed):  
    torch.manual_seed(seed)       # Current CPU  
    torch.cuda.manual_seed(seed)  # Current GPU  
    np.random.seed(seed)          # Numpy module  
    random.seed(seed)             # Python random module  
    torch.backends.cudnn.benchmark = False    # Close optimization  
    torch.backends.cudnn.deterministic = True # Close optimization  
    torch.cuda.manual_seed_all(seed) # All GPU (Optional)


def main():
    # Initialize the parser
    parser = argparse.ArgumentParser(description="arguments for training decoder-only models")

    # Add argument
    parser.add_argument('-n', '--name', type=str, required=True, help="Selected name of model")

    # Parse the arguments
    args = parser.parse_args()

    seed_everything(2023)
    max_token_len = 600
    tokenizer = AutoTokenizer.from_pretrained(args.name, bos_token='<startoftext>', eos_token='<endoftext>', pad_token='<pad>')

    # special_tokens = ['<|disease|>', '<|/disease|>', '<|country|>', '<|/country|>', '<|province|>', '<|/province|>', '<|city|>', '<|/city|>', 
    #         '<|country Code|>', '<|/country Code|>', '<|province Code|>', '<|/province Code|>', '<|city Code|>', '<|/city Code|>', '<|virus|>', 
    #         '<|/virus|>', '<|symptoms|>', '<|/symptoms|>', '<|victims|>', '<|/victims|>', '[None]', '[And]', '<sep>']
    # num_added_tokens = tokenizer.add_special_tokens({'additional_special_tokens': special_tokens})

    train_dataset = EventExtractionDataset("./dataset/finetuned_data/rand_direct_allrole/train_all.pkl", tokenizer, max_length=max_token_len)
    dev_dataset = EventExtractionDataset("./dataset/finetuned_data/rand_direct_allrole/dev_all.pkl", tokenizer, max_length=max_token_len)

    config = AutoConfig.from_pretrained(args.name)
    model = AutoModelForCausalLM.from_pretrained(args.name, config=config)


    model.resize_token_embeddings(len(tokenizer)) 


    training_args = TrainingArguments(
        seed=2023,
        output_dir=f"./output/direct_{args.name.split('/')[-1]}_event_extraction_allrole_test",
        overwrite_output_dir=True,
        num_train_epochs=100,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        evaluation_strategy="steps",
        eval_steps=100,
        save_steps=200,
        save_strategy='steps',
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        learning_rate=1e-5,
        warmup_steps=300,  # Number of warmup steps
        weight_decay=0.01,
    )

    early_stopping_callback = EarlyStoppingCallback(
        early_stopping_patience=5,
        early_stopping_threshold=0.001
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=lambda data: {'input_ids': torch.stack([f[0] for f in data]),
                                    'attention_mask': torch.stack([f[1] for f in data]),
                                    'labels': torch.stack([f[0] for f in data])},
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        callbacks=[early_stopping_callback],
    )

    trainer.train()

if __name__ == "__main__":
    main()
