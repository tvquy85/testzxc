import pandas as pd
import json
import yaml
import torch
import gc
import os
import argparse
from tqdm import tqdm
import requests
from dotenv import load_dotenv
import threading
from concurrent.futures import ThreadPoolExecutor

from transformers import AutoTokenizer, AutoModelForCausalLM
import sys
sys.path.append('src/judges')
from inferability_judge import build_inferability_prompts, parse_inferability_outputs

load_dotenv('d:\\Conferences\\FitAI\\.env')
BASE_URL = os.getenv('OPENAI_API_BASE', 'https://ai-fit.hcmus.edu.vn/openai')
API_KEY = os.getenv('OPENAI_API_KEY')
HEADERS = {
    'Authorization': f'Bearer {API_KEY}', 
    'Content-Type': 'application/json', 
    'User-Agent': 'curl/7.81.0'
}

def generate_rationale_api(prompt):
    url = f'{BASE_URL}/chat/completions'
    payload = {
        'model': 'Qwen3.6-27B', 
        'messages': prompt, 
        'max_tokens': 500,
        'temperature': 0.7,
        'top_p': 0.9
    }
    try:
        r = requests.post(url, headers=HEADERS, json=payload, timeout=60)
        if r.status_code == 200:
            return r.json()['choices'][0]['message']['content']
        else:
            return f"Error: API returned {r.status_code}"
    except Exception as e:
        return f"Error: Exception {e}"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--threads", type=int, default=10)
    args = parser.parse_args()

    with open("configs/local_paths.yaml", 'r') as f:
        paths = yaml.safe_load(f)

    print("1. Loading test set (latest 15% dates)...")
    df_samples = pd.read_parquet("data/labels/aligned_samples_h1.parquet")
    df_tech = pd.read_parquet("data/indicators/technical_event_tokens_h1.parquet")
    
    df = df_samples.merge(df_tech[['sample_id', 'technical_event_tokens']], on='sample_id', how='left')
    df['window_end_date'] = pd.to_datetime(df['window_end_date'])
    df = df.sort_values('window_end_date')
    
    test_size = int(len(df) * 0.15)
    df_test = df.tail(test_size).copy()
    
    if args.limit:
        df_test = df_test.tail(args.limit).copy()
        
    print(f"Test set size: {len(df_test)} samples")
    
    with open("prompts/rationale_generation_prompt.txt", "r", encoding="utf-8") as f:
        prompt_template = f.read()

    generator_prompts = []
    headlines = df_test['headline'].fillna("").tolist()
    tech_tokens = df_test['technical_event_tokens'].fillna("").tolist()
    
    for hl, tt in zip(headlines, tech_tokens):
        p = prompt_template.replace("{headline}", hl).replace("{technical_event_tokens}", tt)
        generator_prompts.append([
            {"role": "system", "content": "You are an elite financial trader."},
            {"role": "user", "content": p}
        ])

    print(f"2. Generating Rationales using FitAI Qwen3.6-27B API with {args.threads} threads...")
    cache_file = args.output + ".cache.json"
    if os.path.exists(cache_file):
        print(f"Loading cached rationales from {cache_file}...")
        with open(cache_file, "r", encoding="utf-8") as f:
            rationales_text = json.load(f)
    else:
        rationales_text = [""] * len(generator_prompts)
        
    from threading import Lock
    save_lock = Lock()
    completed = 0
    
    def process_prompt(idx):
        nonlocal completed
        # Skip if already cached
        if rationales_text[idx] == "":
            rationales_text[idx] = generate_rationale_api(generator_prompts[idx])
        
        with save_lock:
            completed += 1
            if completed % 50 == 0:
                with open(cache_file, "w", encoding="utf-8") as f:
                    json.dump(rationales_text, f)
                    
    with ThreadPoolExecutor(max_workers=args.threads) as executor:
        list(tqdm(executor.map(process_prompt, range(len(generator_prompts))), total=len(generator_prompts), desc="API Generation"))
        
    print(f"Saving final rationales to {cache_file}...")
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(rationales_text, f)

    print("3. Evaluating with Llama-3 Proxy Judge...")
    llama3_path = paths['models']['llama3_judge']
    tokenizer = AutoTokenizer.from_pretrained(llama3_path)
    tokenizer.padding_side = 'left'
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        
    model = AutoModelForCausalLM.from_pretrained(llama3_path, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True)
    
    infer_prompts = build_inferability_prompts(headlines, tech_tokens, rationales_text, tokenizer)
    infer_texts = []
    
    batch_size = 4 
    for i in tqdm(range(0, len(infer_prompts), batch_size), desc="Proxy Judging"):
        batch = infer_prompts[i:i+batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=2048).to(model.device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=256, temperature=0.0, do_sample=False)
        for j in range(len(batch)):
            generated_ids = outputs[j][inputs.input_ids[j].shape[-1]:]
            infer_texts.append(tokenizer.decode(generated_ids, skip_special_tokens=True))
            
    infer_results = parse_inferability_outputs(infer_texts)
    
    del model
    del tokenizer
    torch.cuda.empty_cache()
    gc.collect()
    
    print("4. Saving outputs...")
    results = []
    for idx, row in enumerate(df_test.itertuples()):
        res = infer_results[idx] if idx < len(infer_results) else {}
        item = {
            "sample_id": row.sample_id,
            "window_end_date": str(row.window_end_date),
            "headline": headlines[idx],
            "technical_event_tokens": tech_tokens[idx],
            "true_label": row.label_5,
            "rationale": rationales_text[idx],
            "p_strong_down": res.get('strong_down', 0.0),
            "p_mild_down": res.get('mild_down', 0.0),
            "p_neutral": res.get('neutral', 0.0),
            "p_mild_up": res.get('mild_up', 0.0),
            "p_strong_up": res.get('strong_up', 0.0),
        }
        pred_lbl = max(res, key=res.get) if res else 'neutral'
        lbl_rev_map = {
            'strong_down': 'Strong Down',
            'mild_down': 'Mild Down',
            'neutral': 'Neutral',
            'mild_up': 'Mild Up',
            'strong_up': 'Strong Up'
        }
        item["pred_label"] = lbl_rev_map.get(pred_lbl, 'Neutral')
        results.append(item)
        
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for item in results:
            f.write(json.dumps(item) + "\n")
            
    print(f"Saved {len(results)} test predictions to {args.output}")

if __name__ == "__main__":
    main()
