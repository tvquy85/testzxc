import json
import re
from typing import List, Dict

def build_financial_soundness_prompts(
    headlines: List[str], 
    tech_tokens_list: List[str], 
    rationales: List[str], 
    tokenizer
) -> List[str]:
    with open("prompts/financial_soundness_judge_prompt.txt", "r") as f:
        template = f.read()
        
    prompts = []
    for h, t, r in zip(headlines, tech_tokens_list, rationales):
        user_msg = template.replace("{headline}", str(h)).replace("{technical_event_tokens}", str(t)).replace("{rationale_text}", str(r))
        messages = [
            {"role": "system", "content": "You are a financial soundness judge. Output strictly valid JSON."},
            {"role": "user", "content": user_msg}
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompts.append(prompt)
    return prompts

def parse_financial_soundness_outputs(outputs: List[str]) -> List[Dict]:
    results = []
    default_scores = {
        "financial_soundness": 0.0,
        "groundedness": 0.0,
        "overconfidence": 0.0,
        "main_error": "Failed to parse"
    }
    
    for text in outputs:
        score = default_scores.copy()
        try:
            for key in ["financial_soundness", "groundedness", "overconfidence"]:
                match = re.search(f'"{key}"\\s*:\\s*([0-9.]+)', text)
                if match:
                    score[key] = float(match.group(1))
        except Exception:
            pass
            
        results.append(score)
    return results
