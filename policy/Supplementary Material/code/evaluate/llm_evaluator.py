import pickle
from transformers import GenerationConfig
import torch
from peft import PeftModel    
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer, StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer
from transformers import LlamaForCausalLM
import torch.nn.functional as F

Action = ['DEAD', 'STOP', 'NORTH', 'SOUTH', \
    'EAST', 'WEST', 'Attack Enemy 0', 'Attack Enemy 1', 'Attack Enemy 2', 'Attack Enemy 3', 'Attack Enemy 4', 'Attack Enemy 5']

for k in range(xxx):
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



from openai import OpenAI
client = OpenAI(api_key = "xxx")
def ask_gpt(messages):
   
    completion = client.chat.completions.create(
                    model="gpt-4o", 
                    messages = messages,
                    )
   
    return completion.choices[0].message.content


#explanation llm
model1 = AutoModelForCausalLM.from_pretrained("xxx/model", torch_dtype=torch.bfloat16).cuda()
model1 = PeftModel.from_pretrained(model1, 'xxxx/project/trl/examples/scripts/xxx',torch_dtype=torch.bfloat16).cuda()
model1 = model1.merge_and_unload().cuda()
tokenizer1 = AutoTokenizer.from_pretrained('xxx/model', legacy=False)
tokenizer1.pad_token = tokenizer.eos_token



correct = 0
length = len(policy_text)
for idx in range(length):     
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

    new_response = tokenizer.decode(response[0][1:])+f'Based on the above content, please directly tell me which is the predicted action chosen from {Action}? The response should be in the format: The predicted action is [xxx].'            
  
    messages = [{"role": "user", "content": new_response}]
    sum_text = ask_gpt(messages)
    if f'The predicted action is [{Action[data]}].' in sum_text:
        correct += 1

        
print(f'accuracy: {correct/length}') 
