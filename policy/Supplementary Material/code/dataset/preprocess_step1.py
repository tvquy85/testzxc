#For faster the rectified flow model training, we preprocess the data to get the corresponding the last layer's hidden_states, attention_mask and etc. in the transformer.


from openai import OpenAI
import os
import copy
import numpy as np
import pickle
import random
from collections import defaultdict
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer, StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer, LlamaForCausalLM
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json
import torch.nn.functional as F


model_name = 'llama3'
positive = 'positive'


max_length = 2713 #smac #1024 mmlu #628 mathqa

dataset = torch.load(f'xxx/xxx.pt')


if model_name=='qwen':
    model2 = AutoModelForCausalLM.from_pretrained("xxx/model/qwen/Qwen2.5-7B-Instruct", torch_dtype=torch.bfloat16).cuda()
    lm_backbone = getattr(model2, model2.base_model_prefix).to("cuda")
    tokenizer = AutoTokenizer.from_pretrained('xxx/model/qwen/Qwen2.5-7B-Instruct', legacy=False)
    tokenizer.pad_token = tokenizer.eos_token
    stop_token_ids = [0]
elif model_name=='llama3':
    model2 = AutoModelForCausalLM.from_pretrained("xxx/model/llama3.1-8B/Llama-3.1-8B-Instruct", torch_dtype=torch.bfloat16).cuda()
    lm_backbone = getattr(model2, model2.base_model_prefix).to("cuda")
    tokenizer = AutoTokenizer.from_pretrained('xxx/model/llama3.1-8B/Llama-3.1-8B-Instruct', legacy=False)
    tokenizer.pad_token = tokenizer.eos_token
    stop_token_ids = [0]
else:
    model2 = AutoModelForCausalLM.from_pretrained("xxx/model/gemma/gemma-2-2b-it", torch_dtype=torch.bfloat16).cuda()
    lm_backbone = getattr(model2, model2.base_model_prefix).to("cuda")
    tokenizer = AutoTokenizer.from_pretrained('xxx/model/gemma/gemma-2-2b-it', legacy=False)
    tokenizer.pad_token = tokenizer.eos_token
    stop_token_ids = [0]


if model_name=='gemma':
    if positive=='positive':
        datatexts = {"gt_action":[], "last_layer_hidden_states":[],"attention_mask":[],"action_prob":[]}#,
    else:
        datatexts = {"gt_action":[], "last_layer_hidden_states":[],"attention_mask":[]}#,
else:
    if positive=='positive':
        datatexts = {"gt_action":[], "last_layer_hidden_states":[],"attention_mask":[],"position_embeddings":[],"action_prob":[]}#,
    else:
        datatexts = {"gt_action":[], "last_layer_hidden_states":[],"attention_mask":[],"position_embeddings":[]}#,


length = len(dataset['prompt'])

idx = 0


for i in range(length):
        prompt = dataset['prompt'][i]
        if model_name=='gemma':
            inputs1 = tokenizer(prompt, return_tensors="pt").to("cuda:0")
            input_ids1_part = inputs1["input_ids"].to("cuda:0")
        elif model_name=='qwen':
            inputs1 = tokenizer(prompt, return_tensors="pt").to("cuda:0")
            input_ids1_part = inputs1["input_ids"].to("cuda:0")  
        else:
            inputs1 = tokenizer(prompt, return_tensors="pt").to("cuda:0")
            input_ids1_part = inputs1["input_ids"].to("cuda:0")
            
        if input_ids1_part.shape[1]<max_length:
            input_ids1_part = torch.cat((torch.tensor([[tokenizer.pad_token_id for _ in range(max_length-input_ids1_part.shape[1])]]).to("cuda:0"),input_ids1_part),dim=1)
        else:
            input_ids1_part = input_ids1_part[:,input_ids1_part.shape[1]-max_length:]
        
        attention_mask =  input_ids1_part != tokenizer.pad_token_id    
        input_ids1_part = torch.masked_fill(input_ids1_part, ~attention_mask, 0)
        position_ids = attention_mask.cumsum(1) - attention_mask.long()
        if i ==  length_v5-1 :
            save_flag = True
        else:
            save_flag = False           
             
        with torch.no_grad():
            h = lm_backbone(
                input_ids=input_ids1_part,
                position_ids=position_ids,
                attention_mask=attention_mask,
                return_dict=True,
                output_hidden_states=True,
                use_cache=False,  # otherwise mistral-based RM would error out
                save_flag = save_flag)
        
        del h, input_ids1_part, attention_mask
