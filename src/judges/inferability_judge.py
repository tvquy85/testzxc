import json
import re
from typing import List, Dict

def build_inferability_prompts(
    headlines: List[str], 
    tech_tokens_list: List[str], 
    rationales: List[str], 
    tokenizer
) -> List[str]:
    with open("prompts/proxy_inferability_judge_prompt.txt", "r") as f:
        template = f.read()
        
    prompts = []
    for h, t, r in zip(headlines, tech_tokens_list, rationales):
        user_msg = template.replace("{headline}", str(h)).replace("{technical_event_tokens}", str(t)).replace("{rationale_text}", str(r))
        messages = [
            {"role": "system", "content": "You are a proxy inferability judge. Output valid JSON."},
            {"role": "user", "content": user_msg}
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompts.append(prompt)
    return prompts

def parse_inferability_outputs(outputs: List[str]) -> List[Dict]:
    results = []
    default_dist = {
        "strong_down": 0.0,
        "mild_down": 0.0,
        "neutral": 1.0,
        "mild_up": 0.0,
        "strong_up": 0.0
    }
    
    for text in outputs:
        dist = default_dist.copy()
        try:
            for key in default_dist.keys():
                match = re.search(f'"{key}"\\s*:\\s*([0-9.]+)', text)
                if match:
                    dist[key] = float(match.group(1))
        except Exception:
            pass
            
        total = sum(dist.values())
        if total > 0:
            dist = {k: v / total for k, v in dist.items()}
        else:
            dist = default_dist.copy()
            
        results.append(dist)
    return results
