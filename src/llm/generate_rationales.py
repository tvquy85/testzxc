import os
import json
import argparse
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import hashlib
from src.llm.render_context import render_context
from src.llm.parse_and_validate_rationale import parse_llm_json, validate_rationale_schema

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=str, default="data/labels/aligned_samples_h1.parquet")
    parser.add_argument("--tech", type=str, default="data/indicators/technical_event_tokens_h1.parquet")
    parser.add_argument("--prompt", type=str, default="prompts/rationale_generation_prompt.txt")
    parser.add_argument("--config", type=str, default="configs/local_paths.yaml")
    parser.add_argument("--limit", type=int, default=5000)
    parser.add_argument("--num-candidates", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--output", type=str, default="data/rationales/candidate_rationales_h1.jsonl")
    args = parser.parse_args()

    import yaml
    with open(args.config, 'r') as f:
        paths = yaml.safe_load(f)
    model_name = paths.get('models', {}).get('main_explanation_llm')
    print(f"Loading model from {model_name}...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # We load in 4-bit or 8-bit to save VRAM and maybe faster? Actually 16-bit is fine on 24GB for 3B.
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype=torch.float16, 
        device_map="auto",
        trust_remote_code=True
    )
    
    with open(args.prompt, "r") as f:
        prompt_template = f.read()
        
    prompt_hash = hashlib.md5(prompt_template.encode()).hexdigest()

    samples = pd.read_parquet(args.samples)
    tech = pd.read_parquet(args.tech)
    df = pd.merge(samples, tech[["sample_id", "technical_event_tokens", "regime_label"]], on="sample_id")
    
    if args.limit:
        df = df.sample(n=min(len(df), args.limit), random_state=42)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    results = []
    schema_oks = 0
    
    print("Generating rationales...")
    batch_size = 4
    
    # Pre-render prompts
    all_texts = []
    sample_ids = []
    for idx, row in df.iterrows():
        ctx = render_context(row)
        prompt_text = prompt_template
        for k, v in ctx.items():
            prompt_text = prompt_text.replace(f"{{{k}}}", str(v))
            
        messages = [
            {"role": "system", "content": "You are a financial analyst."},
            {"role": "user", "content": prompt_text}
        ]
        text_input = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        # Replicate for candidates
        for c in range(args.num_candidates):
            all_texts.append(text_input)
            sample_ids.append((row["sample_id"], c))
            
    with open(args.output, "w", encoding="utf-8") as out_f:
        for i in tqdm(range(0, len(all_texts), batch_size)):
            batch_texts = all_texts[i:i+batch_size]
            batch_ids = sample_ids[i:i+batch_size]
            
            inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=2048).to(device)
            with torch.no_grad():
                outputs = model.generate(
                    **inputs, 
                    max_new_tokens=args.max_new_tokens, 
                    temperature=args.temperature,
                    do_sample=True
                )
            
            for j in range(len(batch_texts)):
                generated_ids = outputs[j][inputs.input_ids[j].shape[-1]:]
                output_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
                
                parsed_json = parse_llm_json(output_text)
                parse_ok = parsed_json is not None
                schema_ok = validate_rationale_schema(parsed_json) if parse_ok else False
                
                if schema_ok: schema_oks += 1
                
                res = {
                    "sample_id": batch_ids[j][0],
                    "candidate_id": batch_ids[j][1],
                    "generator_model": model_name,
                    "prompt_hash": prompt_hash,
                    "rationale_json": parsed_json,
                    "raw_text": output_text,
                    "parse_ok": parse_ok,
                    "schema_ok": schema_ok
                }
                out_f.write(json.dumps(res) + "\n")
                out_f.flush()
                results.append(res)
                
    status = {
        "step": "08_GENERATE_RATIONALES_LOCAL_LLM",
        "status": "PASS" if schema_oks / max(1, len(results)) >= 0.7 and len(results) >= 1000 else "FAIL",
        "candidate_count": len(results),
        "unique_sample_count": len(df),
        "schema_ok_rate": schema_oks / max(1, len(results)),
        "generator_model": model_name,
        "notes": "Generated candidates with local LLM"
    }
    
    with open("outputs/status/08_GENERATE_RATIONALES_LOCAL_LLM.status.json", "w") as f:
        json.dump(status, f, indent=2)
        
    print(json.dumps(status, indent=2))

if __name__ == "__main__":
    main()
