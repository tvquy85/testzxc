from openai import OpenAI
import os
import copy
import numpy as np
import pickle
import random
from collections import defaultdict
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json
import torch.nn.functional as F
policy_text = []
chosen_answer_text = []
rejected_answer_text = []
action_text = []

Action = ['DEAD', 'STOP', 'NORTH', 'SOUTH', \
    'EAST', 'WEST', 'Attack Enemy 0', 'Attack Enemy 1', 'Attack Enemy 2', 'Attack Enemy 3', 'Attack Enemy 4', 'Attack Enemy 5']
lower_action = ['dead', 'stop', 'north', 'south', \
    'east', 'west', 'attack enemy 0', 'attack enemy 1', 'attack enemy 2', 'attack enemy 3', 'attack enemy 4', 'attack enemy 5']
chose = 'seed'

for k in range(xxx):
    
    with open('xxx/dataset/test/traj_'+ chose+f'{k}'+'.pkl', 'rb') as file:
        loaded_data = pickle.load(file)
    with open('xxx/dataset/test/value_'+ chose+f'{k}'+'.pkl', 'rb') as file:
        loaded_value_data = pickle.load(file)
    
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
                for j in range(6):#3
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
                    rejected_text.append(random.choice(copy_action))
                
                else:
                    agent_text[i] += f'Action: {Action[data]}; '
                    agent_text[i] += f'Value: {np.float16(loaded_value_data[int(t//2)][i])}'
                    agent_text[i] += '} '

client = OpenAI(api_key = "xxx")

def ask_gpt(messages):
   
    completion = client.chat.completions.create(
                    model="o1-mini",
                    messages = messages,
                   
                    )
   
    return completion.choices[0].message.content


data = []
test_data = []

for idx in range(xxx):     
      
    prompt = 'Human: ' + f'Here are 5 allies and 6 enemies in SMAC. The goal of each ally is to attack all the enemies together with other allies. Every step, each ally can choose one action to take \
        form the Action set {Action}. '+'The trajectory of each ally consisists of a series of {State, Action, Value} pairs, where an ally takes Action based on the State and transits \
        to the next State. A higher Value indicates a better State transition. Here is 4 consecutive {State, Action, Value} pairs where the last state-action pair misses the Action: ' \
        + policy_text[idx]+ f'Please help me concisely summarize this part of the trajectory and predict the missing Action. '
       
     
    messages = [{"role": "user", "content": prompt}]
    sum_text = ask_gpt(messages)
    
    save_dir = f'xxx.pkl'
    pickle.dump(sum_text, open(save_dir, 'wb'))