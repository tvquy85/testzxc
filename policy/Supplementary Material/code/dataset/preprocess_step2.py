#preprocess the data to transforms hidden_states, attention_mask and etc. to a single .pt file for data loading during the rectified flow training and evaluation.

from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer, StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer, LlamaForCausalLM
import pickle

model_name = 'llama3'#gemma #qwen
positive = 'positive'#negative

with open(f'xxx/{model_name}_{positive}_last_layer_hidden_states.pkl', 'rb') as file:
    last_layer_hidden_states = pickle.load(file) 

    
with open(f'xxx/{model_name}_{positive}_attention_mask.pkl', 'rb') as file:
    attention_mask = pickle.load(file) 

if model_name=='qwen' or  model_name=='llama3':
    all_position_embeddings = []
    with open(f'xxx/{model_name}_{positive}_position_embeddings.pkl', 'rb') as file:
        position_embeddings = pickle.load(file) 
        for i in range(len(position_embeddings)):
            all_position_embeddings.append([])
            for j in range(len(position_embeddings[i])):
                all_position_embeddings[i].append(position_embeddings[i][j].float().cpu().tolist())


max_length = 2713 #llama #1024 mmlu #628 mathqa

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
        datatexts = {"gt_action":[], "last_layer_hidden_states":[],"attention_mask":[]}
else:
    if positive=='positive':
        datatexts = {"gt_action":[], "last_layer_hidden_states":[],"attention_mask":[],"position_embeddings":[],"action_prob":[]}#,
    else:
        datatexts = {"gt_action":[], "last_layer_hidden_states":[],"attention_mask":[],"position_embeddings":[]}#,

length = len(dataset['prompt'])

for i in range(length):
          
    datatexts["gt_action"].append(dataset["gt_action"][i])
    
    if positive=='positive':
        datatexts["action_prob"].append(dataset["action_prob"][i])
    datatexts["last_layer_hidden_states"].append(last_layer_hidden_states[i])
    datatexts["attention_mask"].append(attention_mask[i])
    if model_name=='qwen'or model_name=='llama3':
        datatexts["position_embeddings"].append(torch.tensor(all_position_embeddings[i]))

if positive=='positive':
    torch.save(datatexts,f'xxx/{model_name}_train.pt')
else:
    torch.save(datatexts,f'xxx/{model_name}_test_negative.pt')
