import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, GPT2Config
import pickle
import os


os.environ['CUDA_VISIBLE_DEVICES']="1"

def generate_event(model, tokenizer, document):
    # input_text = f"<endoftext> Document: {document} <sep> Event: "
    prompt = "An outbreak event includes the following attributes: disease, city, province, country, city geocode, province geocode (e.g. Ohio has geocode 5165418), country geocode (e.g. United Kingdom has geocode 2635167), virus/bacteria (named as virus), symptoms (e.g. coughing), and victims (one of Human, Animal, or Plant). Note that geocodes are not in the text, so you'll need to use your background knowledge to infer them. If some attributes are not explicitly mentioned, try to infer them carefully. If the attributes cannot be found or inferred from the text, return unknown. Provide your answer in JSON format. Multiple values can be provided for the same attribute if necessary. If one attribute has multiple values, return them separated with ';'."
    input_text = f'<startoftext> Document: {document} <sep> Prompt: {prompt} <sep> Event: '

    input_ids = tokenizer.encode(input_text, return_tensors="pt", add_special_tokens=False, truncation=True, max_length=600, padding=False)
    output = model.generate(input_ids.cuda(), max_length=800, do_sample=True, top_k=4, eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id, early_stopping=True)
    # output = model.generate(input_ids.cuda(), max_length=800, num_beams=5, temperature=0.7, no_repeat_ngram_size=3, eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id, early_stopping=True)

    generated_text = tokenizer.decode(output[0])

    generated_event = generated_text[len(input_text):].strip()
    return generated_event

def main():
    # model_name = "gpt2"
    # model_path = "./output/direct_gpt2_event_extraction_allrole_new/checkpoint-3800"
    # save_to = "./predictions/evaluation_gpt2_new.json"

    model_name = "facebook/opt-125m"
    model_path = "./output/direct_template_opt_event_extraction_allrole_new/checkpoint-3200"
    save_to = "./predictions/evaluation_opt_tmp.json"

    # model_name = "bigscience/bloom-560m"
    # model_path = "./output/direct_bloom_event_extraction_allrole_new/checkpoint-2100"
    # save_to = "./predictions/evaluation_bloom_new.json"

    # model_name = "facebook/galactica-125m"
    # model_path = "./output/direct_galactica_event_extraction_allrole_new/checkpoint-3200"
    # save_to = "./predictions/evaluation_galactica_new.json"

    # model_name = "EleutherAI/gpt-neo-125m"
    # model_path = "./output/direct_gptneo_event_extraction_allrole3/checkpoint-8000"
    # save_to = "./predictions/evaluation_gptneo3.json"

    tokenizer = AutoTokenizer.from_pretrained(model_name, bos_token='<startoftext>', eos_token='<endoftext>', pad_token='<pad>')
    # tokenizer = AutoTokenizer.from_pretrained(model_name, pad_token='<pad>')

    # special_tokens = ['<|disease|>', '<|/disease|>', '<|country|>', '<|/country|>', '<|province|>', '<|/province|>', '<|city|>', '<|/city|>', 
    #         '<|country Code|>', '<|/country Code|>', '<|province Code|>', '<|/province Code|>', '<|city Code|>', '<|/city Code|>', '<|virus|>', 
    #         '<|/virus|>', '<|symptoms|>', '<|/symptoms|>', '<|victims|>', '<|/victims|>', '[None]', '[And]', '<sep>']
    # num_added_tokens = tokenizer.add_special_tokens({'additional_special_tokens': special_tokens})

    # config = GPT2Config.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained(model_path)

    
    model.resize_token_embeddings(len(tokenizer))
    model.cuda()
    model.eval()

    file_path = "./dataset/finetuned_data/rand_direct_allrole/test_all.pkl"
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
        eval_dataset = []
        for l_in, l_out, l_label in zip(data['input'], data['target'], data['all']):
            eval_dataset.append({'document': l_in, 'event': l_out})

    results = []

    for idx in range(len(eval_dataset)):
        document, true_event = eval_dataset[idx].values()
        generated_event = generate_event(model, tokenizer, document)
        
        results.append({
            "document": document,
            "true_event": true_event,
            "generated_event": generated_event
        })

    with open(save_to, "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
