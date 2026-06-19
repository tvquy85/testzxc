import pandas as pd
import json
import time
import os
import requests
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

load_dotenv('d:\\Conferences\\FitAI\\.env')
BASE_URL = os.getenv('OPENAI_API_BASE', 'https://ai-fit.hcmus.edu.vn/openai')
API_KEY = os.getenv('OPENAI_API_KEY')
HEADERS = {
    'Authorization': f'Bearer {API_KEY}', 
    'Content-Type': 'application/json',
    'User-Agent': 'curl/7.81.0'
}

def generate_rationale_api(prompt, max_tokens=500):
    url = f'{BASE_URL}/chat/completions'
    payload = {
        'model': 'Qwen3.6-27B', 
        'messages': prompt, 
        'max_tokens': max_tokens,
        'temperature': 0.7,
        'top_p': 0.9
    }
    try:
        r = requests.post(url, headers=HEADERS, json=payload, timeout=120)
        if r.status_code == 200:
            return r.json()['choices'][0]['message']['content']
        else:
            return f"Error: API returned {r.status_code}"
    except Exception as e:
        return f"Error: Exception {e}"

def test_single(df, max_workers=10):
    with open("prompts/rationale_generation_prompt.txt", "r", encoding="utf-8") as f:
        prompt_template = f.read()

    prompts = []
    for row in df.itertuples():
        p = prompt_template.replace("{headline}", row.headline).replace("{technical_event_tokens}", row.technical_event_tokens)
        if "{body}" in p:
            p = p.replace("{body}", str(row.body))
        if "{regime_label}" in p:
            p = p.replace("{regime_label}", "Unknown")
            
        prompts.append([
            {"role": "system", "content": "You are an elite financial trader."},
            {"role": "user", "content": p}
        ])

    start_time = time.time()
    results = []
    
    def process(p):
        return generate_rationale_api(p, max_tokens=500)
        
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(tqdm(executor.map(process, prompts), total=len(prompts), desc="Single Generation"))
        
    elapsed = time.time() - start_time
    return results, elapsed

def test_batch(df, batch_size=5, max_workers=10):
    with open("prompts/rationale_batch_generation_prompt.txt", "r", encoding="utf-8") as f:
        prompt_template = f.read()

    batches = []
    for i in range(0, len(df), batch_size):
        batch_df = df.iloc[i:i+batch_size]
        samples_list = []
        for row in batch_df.itertuples():
            samples_list.append({
                "sample_id": row.sample_id,
                "headline": row.headline,
                "technical_event_tokens": row.technical_event_tokens
            })
            
        p = prompt_template.replace("{batch_size}", str(len(samples_list)))
        p = p.replace("{samples_json}", json.dumps(samples_list, indent=2))
        
        batches.append([
            {"role": "system", "content": "You are an elite financial trader."},
            {"role": "user", "content": p}
        ])

    start_time = time.time()
    results = []
    
    def process(p):
        return generate_rationale_api(p, max_tokens=2500) # Increased max_tokens for multiple outputs
        
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(tqdm(executor.map(process, batches), total=len(batches), desc="Batch Generation"))
        
    elapsed = time.time() - start_time
    return results, elapsed

def main():
    print("Loading test data...")
    df_samples = pd.read_parquet("data/labels/aligned_samples_h1.parquet")
    df_tech = pd.read_parquet("data/indicators/technical_event_tokens_h1.parquet")
    df = df_samples.merge(df_tech[['sample_id', 'technical_event_tokens']], on='sample_id', how='left')
    df['window_end_date'] = pd.to_datetime(df['window_end_date'])
    df = df.sort_values('window_end_date')
    
    test_size = int(len(df) * 0.15)
    df_test = df.tail(test_size).head(100).copy() # Take first 100 of the test set
    
    print(f"Testing with {len(df_test)} samples.")
    
    print("\n--- Running Baseline (Single Prompting) ---")
    single_results, single_time = test_single(df_test, max_workers=10)
    print(f"Baseline Time: {single_time:.2f} seconds ({len(df_test)/single_time:.2f} samples/s)")
    
    print("\n--- Running Batched (K=5) Prompting ---")
    batch_results, batch_time = test_batch(df_test, batch_size=5, max_workers=10)
    print(f"Batched Time: {batch_time:.2f} seconds ({len(df_test)/batch_time:.2f} samples/s)")
    
    # Validation
    valid_batch_items = 0
    total_batch_items = 0
    for raw_out in batch_results:
        try:
            # Clean up potential markdown formatting like ```json
            out = raw_out.strip()
            if out.startswith("```json"): out = out[7:]
            if out.startswith("```"): out = out[3:]
            if out.endswith("```"): out = out[:-3]
            out = out.strip()
            
            parsed = json.loads(out)
            if isinstance(parsed, list):
                valid_batch_items += len(parsed)
                total_batch_items += 5 # We requested 5
        except Exception as e:
            total_batch_items += 5
            pass
            
    print(f"\nBatch JSON Parse Success Rate: {valid_batch_items}/{total_batch_items} items")
    
    with open("outputs/test_batch_prompt_results.json", "w") as f:
        json.dump({"single": single_results, "batched": batch_results}, f, indent=2)
        
    print("Results saved to outputs/test_batch_prompt_results.json")

if __name__ == "__main__":
    main()
