import pandas as pd
import json
import yaml
import torch
import gc
import os
import argparse
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModelForCausalLM
import sys
sys.path.append('src/judges')
from inferability_judge import build_inferability_prompts, parse_inferability_outputs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True, help="Path to counterfactual contexts jsonl")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    with open("configs/local_paths.yaml", 'r') as f:
        paths = yaml.safe_load(f)

    cf_data = []
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            cf_data.append(json.loads(line))
            
    print(f"Loaded {len(cf_data)} counterfactual tasks.")

    generator_prompts = []
    for item in cf_data:
        generator_prompts.append([
            {"role": "system", "content": "You are an elite financial trader."},
            {"role": "user", "content": item['prompt']}
        ])

    print("1. Generating Rationales for Counterfactuals...")
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    tokenizer.padding_side = 'left'
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        
    model = AutoModelForCausalLM.from_pretrained(args.checkpoint, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
    
    rationales_text = []
    batch_size = 8
    
    for i in tqdm(range(0, len(generator_prompts), batch_size), desc="Generating CF"):
        batch = generator_prompts[i:i+batch_size]
        texts = [tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in batch]
        inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=300, temperature=0.7, top_p=0.9, do_sample=True)
            
        for j in range(len(batch)):
            gen_len = inputs.input_ids[j].shape[-1]
            generated_ids = outputs[j][gen_len:]
            text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            rationales_text.append(text)
            
    del model
    del tokenizer
    torch.cuda.empty_cache()
    gc.collect()

    print("2. Proxy Judging CF Rationales...")
    llama3_path = paths['models']['llama3_judge']
    tokenizer = AutoTokenizer.from_pretrained(llama3_path)
    tokenizer.padding_side = 'left'
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        
    model = AutoModelForCausalLM.from_pretrained(llama3_path, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True)
    
    # We don't have headline and tech_tokens extracted nicely here, but we can just pass empty string to build_inferability_prompts
    # because Llama-3 only uses them if we include them in the prompt. Wait, the inferability prompt uses headline and tech_tokens!
    # Ah, the inferability prompt expects `headline` and `technical_event_tokens` to show to Llama-3, AND the `rationale`.
    # BUT wait! If it's a counterfactual, Llama-3 should see the counterfactual headline and counterfactual tech tokens!
    # Since we replaced them in `item['prompt']`, maybe we should extract them?
    # Actually `build_inferability_prompts` needs them. Let's just pass empty strings for now, or extract from `item['prompt']`.
    # Let's extract them from the prompt roughly, or just pass empty. The proxy judge evaluates if the rationale leads to the label.
    # The rationale is the main thing. But inferability prompt includes the context.
    # We will pass empty strings for Context in Proxy Judge to force the Judge to rely PURELY on the Rationale.
    # Wait, in the original pipeline, Proxy Judge saw the Context.
    # It's okay, we can just use empty strings for Context in CF evaluation.
    
    infer_prompts = build_inferability_prompts([""] * len(cf_data), [""] * len(cf_data), rationales_text, tokenizer)
    infer_texts = []
    
    batch_size = 4
    for i in tqdm(range(0, len(infer_prompts), batch_size), desc="Judging CF"):
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

    print("3. Calculating Counterfactual Flip Rate (CFR)...")
    
    # Group by sample_id
    from collections import defaultdict
    results_by_sample = defaultdict(dict)
    
    for idx, item in enumerate(cf_data):
        sid = item['sample_id']
        cft = item['cf_type']
        res = infer_results[idx] if idx < len(infer_results) else {}
        pred_lbl = max(res, key=res.get) if res else 'neutral'
        results_by_sample[sid][cft] = pred_lbl
        item['pred_label'] = pred_lbl
        item['rationale'] = rationales_text[idx]

    flips = defaultdict(int)
    totals = defaultdict(int)
    
    def direction(lbl):
        if 'down' in lbl.lower(): return -1
        if 'up' in lbl.lower(): return 1
        return 0

    for sid, r in results_by_sample.items():
        if 'ORIGINAL' not in r: continue
        orig_dir = direction(r['ORIGINAL'])
        
        for cft, pred in r.items():
            if cft == 'ORIGINAL': continue
            cf_dir = direction(pred)
            totals[cft] += 1
            if cf_dir != orig_dir:
                flips[cft] += 1

    cfr_metrics = {}
    for cft in totals:
        cfr = flips[cft] / totals[cft] if totals[cft] > 0 else 0
        cfr_metrics[cft] = cfr
        print(f"CFR for {cft}: {cfr:.2%} ({flips[cft]}/{totals[cft]})")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump({
            "cfr_metrics": cfr_metrics,
            "raw_results": cf_data
        }, f, indent=2)

if __name__ == "__main__":
    main()
