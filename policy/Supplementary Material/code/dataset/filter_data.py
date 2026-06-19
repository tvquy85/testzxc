#this file is used for filtering the positive data for training the rectified flow model, 
#this is an example for smac dataset.

import pickle
from transformers import GenerationConfig
import torch
from peft import PeftModel    
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer, StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer
from transformers import LlamaForCausalLM
import torch.nn.functional as F



Action = ['DEAD', 'STOP', 'NORTH', 'SOUTH', \
    'EAST', 'WEST', 'Attack Enemy 0', 'Attack Enemy 1', 'Attack Enemy 2', 'Attack Enemy 3', 'Attack Enemy 4', 'Attack Enemy 5']
lower_action = ['dead', 'stop', 'north', 'south', \
    'east', 'west', 'attack enemy 0', 'attack enemy 1', 'attack enemy 2', 'attack enemy 3', 'attack enemy 4', 'attack enemy 5']


policy_text = []
chosen_answer_text = []
rejected_answer_text = []

for k in range(xxx): #'xxx' denotes trajectory length for the collected data
    with open('xxx/dataset/test/traj_seed'+ chose+f'{k}'+'.pkl', 'rb') as file:
        loaded_data = pickle.load(file)
    agent_text = ['','','','','']
    text = ''
    chosen_text = []
    rejected_text = []
    
    for t, datas in enumerate(loaded_data):
        if t!=0 and t % 8 == 0:               
            policy_text += agent_text
            chosen_answer_text += chosen_text
            rejected_answer_text += rejected_text
            chosen_text = []
            rejected_text = []
            agent_text = ['','','','','']
            text = ''
        if t%2 == 0:
            for i, data in enumerate(datas):
                text += '{'
                text += f'State:'+ '{'
                ally_dim = 20 
                vis = 0
                for j in range(4):
                    ally_fea = data[ally_dim*j:ally_dim*(j+1)].astype(np.float16)
                    if ally_fea[0]:
                        vis += 1
                        if vis == 1:
                            text += f'visible allies: '+'{relative (x,y); weapon_cooldown; health} '
                        text += '{'+ f'id {j}:{ally_fea[2:4]}; {ally_fea[4]}; {ally_fea[5]}'
                        text += '}; '
                if vis == 0:
                    text += 'None visible allies; '
                enemy_dim = 8
                vis = 0
                for j in range(6):
                    enemy_fea = data[ally_dim*4+enemy_dim*j:ally_dim*4+enemy_dim*(j+1)].astype(np.float16)
                    if enemy_fea[4]:
                        vis += 1
                        if vis == 1:
                            text += f'visible enemies: '+ '{available attack;  relative (x,y); health} '
                        text += '{'+ f'id {j}: {enemy_fea[0]}; {enemy_fea[2:4]}; {enemy_fea[5]}'
                        text += '}; '
                if vis == 0:
                    text += 'None visible enemies; '
                own_dim = 19 
                own_fea = data[ally_dim*4+enemy_dim*6:ally_dim*4+enemy_dim*6+own_dim].astype(np.float16)
                text += f'own_health: {own_fea[4]}'
                text += '}; '
                agent_text[i] += text
                text = ' '
        else:
            for i, data in enumerate(datas):
                if t % 8 == 7:
                    chosen_text.append(data)
                    copy_action = copy.deepcopy(Action)
                    copy_action.pop(data)
                else:
                    agent_text[i] += f'Action: {Action[data]}; '
                    agent_text[i] += '} '

def logprobs_from_logits(logits: torch.Tensor, labels: torch.Tensor, gather: bool = True) -> torch.Tensor:
    logp = F.log_softmax(logits, dim=2)
    if not gather:
        return logp
    logpy = torch.gather(logp, 2, labels.unsqueeze(2)).squeeze(-1)
    return logpy

#explanation llm
model1 = AutoModelForCausalLM.from_pretrained("xxx/model", torch_dtype=torch.bfloat16).cuda() #'xxx' denotes environment paths for the model.
model1 = PeftModel.from_pretrained(model1, 'xxxx/project/trl/examples/scripts/xxx',torch_dtype=torch.bfloat16).cuda() #'xxx' denotes environment paths for the model.
model1 = model1.merge_and_unload().cuda() 
tokenizer1 = AutoTokenizer.from_pretrained('xxx/model', legacy=False) #'xxx' denotes environment paths for the model.
tokenizer1.pad_token = tokenizer.eos_token

#proxy llms
model21 = AutoModelForCausalLM.from_pretrained("xxx/model", torch_dtype=torch.bfloat16).cuda() #'xxx' denotes environment paths for the model.
# model22 = AutoModelForCausalLM.from_pretrained("xxx/model", torch_dtype=torch.bfloat16).cuda() #'xxx' denotes environment paths for the model.
# model23 = AutoModelForCausalLM.from_pretrained("xxx/model", torch_dtype=torch.bfloat16).cuda() #'xxx' denotes environment paths for the model.
tokenizer21 = AutoTokenizer.from_pretrained('xxx/model', legacy=False) #'xxx' denotes environment paths for the model.
tokenizer21.pad_token = tokenizer.eos_token
# tokenizer22 = AutoTokenizer.from_pretrained('xxx/model', legacy=False) #'xxx' denotes environment paths for the model.
# tokenizer22.pad_token = tokenizer.eos_token
# tokenizer23 = AutoTokenizer.from_pretrained('xxx/model', legacy=False) #'xxx' denotes environment paths for the model.
# tokenizer23.pad_token = tokenizer.eos_token


Action_a = ['DEAD.', 'STOP.', 'NORTH.', 'SOUTH.', \
    'EAST.', 'WEST.', 'Attack Enemy 0.', 'Attack Enemy 1.', 'Attack Enemy 2.',\
     'Attack Enemy 3.', 'Attack Enemy 4.', 'Attack Enemy 5.']
action_tokenized = tokenizer21(Action_a, return_tensors="pt", padding=True)# )

action_length  = [2,2,3,3,2,3,6,6,6,6,6,6]#model_action_length

prompt2 = ["Therefore, the predicted action is"]
inputs2 = tokenizer21(prompt2, return_tensors="pt").to("cuda")
input_ids2 = inputs2["input_ids"][:,1:].to("cuda")


positive_data ={'prompt':[],'gt_action':[],'action_prob':[]}
negative_data = {'prompt':[],'gt_action':[],}
for idx in range(xxx):    #'xxx' denotes trajectory length for the collected data  
    prompt = 'Human: ' + f'Here are 5 allies and 6 enemies in SMAC. The goal of each ally is to attack all the enemies together with other allies. Every step, each ally can choose one action to take \
        form the Action set {Action}. '+'The trajectory of each ally consisists of a series of {State, Action, Value} pairs, where an ally takes Action based on the State and transits \
        to the next State. A higher Value indicates a better State transition. Here is 4 consecutive {State, Action, Value} pairs where the last state-action pair misses the Action: ' \
        + policy_text[idx]+ f'Please help me concisely summarize this part of the trajectory and predict the missing Action. '
       
    inputs1 = tokenizer1(prompt, return_tensors="pt").to("cuda:0")
    input_ids1 = inputs1["input_ids"].to("cuda:0")
    attention_mask1 = inputs1["attention_mask"].to("cuda")      
            
    with torch.no_grad():
        response = model1.generate(
                    input_ids = input_ids1,
                    attention_mask = attention_mask1,
                    max_new_tokens=500,
                    do_sample=True,
                    top_k = 0.0, 
                    top_p = 1.0, 
                    pad_token_id=tokenizer.pad_token_id,
                    return_legacy_cache=True
                )
        reward = []  
        for j in range(len(Action)):
                input_ids_prompt = torch.cat((response,input_ids2.to("cuda"),torch.tensor([(action_tokenized["input_ids"][j][-action_length[j]-1:]).tolist() ]).to("cuda")), dim = 1)
                attention_mask =  input_ids_prompt != tokenizer.pad_token_id           
                input_ids = torch.masked_fill(input_ids_prompt, ~attention_mask, 0)
                logits = model21(input_ids = input_ids, attention_mask = attention_mask)
                logprobs = logprobs_from_logits(logits.logits[:,:-1], input_ids[:,1:])    
                reward.append(logprobs[:,-1-action_length[j]:-1].mean(1,keepdim=True).tolist())

        prompt = tokenizer1.decode(response[0][1:])
        if torch.argmax(torch.tensor(reward)) == int(chosen_answer_text[idx]):
                positive_data['prompt'].append(prompt)
                positive_data['action_prob'].append(torch.tensor(reward))
                positive_data['gt_action'].append(int(chosen_answer_text[idx]))
        else:
                negative_data['prompt'].append(prompt)
                negative_data['gt_action'].append(int(chosen_answer_text[idx]))
        
torch.save(positive_data,'xxx.pt') #'xxx' denotes environment paths for data storage.
torch.save(negative_data,'xxx.pt')   
