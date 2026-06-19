import argparse
import pandas as pd
import json
import os
import sys
sys.path.append(".")
import yaml
from transformers import AutoTokenizer
from src.llm.render_context import render_context

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rationales", required=True)
    parser.add_argument("--flow-rewards", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    
    with open("configs/local_paths.yaml", "r") as f:
        paths = yaml.safe_load(f)
    model_name = paths.get('models', {}).get('main_explanation_llm')
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        
    with open("prompts/rationale_generation_prompt.txt", "r", encoding="utf-8") as f:
        prompt_template = f.read()
        
    df_rat = pd.read_json(args.rationales, lines=True)
    df_rew = pd.read_parquet(args.flow_rewards)
    
    df_samples = pd.read_parquet("data/labels/aligned_samples_h1.parquet")
    df_tech = pd.read_parquet("data/indicators/technical_event_tokens_h1.parquet")
    df_ctx = pd.merge(df_samples, df_tech[["sample_id", "technical_event_tokens", "regime_label"]], on="sample_id")
    
    df = pd.merge(df_rat, df_rew[['sample_id', 'candidate_id', 'flow_overall_reward']], on=['sample_id', 'candidate_id'], how='inner')
    df = df.dropna(subset=['raw_text', 'flow_overall_reward'])
    
    rwsft_data = []
    
    grouped = df.groupby('sample_id')
    for sample_id, group in grouped:
        group = group.sort_values('flow_overall_reward', ascending=False)
        # Take top 2 candidates if their rewards are positive or decent.
        # But for RWSFT, we can just take the top-1 and top-2
        top_cands = group.head(2)
        
        row_ctx = df_ctx[df_ctx['sample_id'] == sample_id].iloc[0]
        ctx_vars = render_context(row_ctx)
        prompt_text = prompt_template
        for k, v in ctx_vars.items():
            prompt_text = prompt_text.replace(f"{{{k}}}", str(v))
            
        # SFT trainer uses conversational format or standard texts
        # Here we format as a standard conversation
        for idx, cand in top_cands.iterrows():
            messages = [
                {"role": "system", "content": "You are a financial analyst."},
                {"role": "user", "content": prompt_text},
                {"role": "assistant", "content": cand['raw_text']}
            ]
            
            # Use flow reward as weight. Make sure it's non-negative.
            # Reward is in roughly [-0.1, 0.8]. We can scale or shift it.
            # We shift by a baseline, e.g., 0.1
            weight = max(0.01, float(cand['flow_overall_reward']) + 0.1)
            
            rwsft_data.append({
                "sample_id": sample_id,
                "messages": messages,
                "weight": weight,
                "flow_reward": float(cand['flow_overall_reward'])
            })
            
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for p in rwsft_data:
            f.write(json.dumps(p) + "\n")
            
    print(f"Created {len(rwsft_data)} RWSFT examples. Saved to {args.output}")

if __name__ == "__main__":
    main()
