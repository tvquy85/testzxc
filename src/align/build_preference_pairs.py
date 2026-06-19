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
    parser.add_argument("--min-margin", type=float, default=0.10)
    args = parser.parse_args()
    
    with open("configs/local_paths.yaml", "r") as f:
        paths = yaml.safe_load(f)
    model_name = paths.get('models', {}).get('main_explanation_llm')
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        
    with open("prompts/rationale_generation_prompt.txt", "r", encoding="utf-8") as f:
        prompt_template = f.read()
        
    # Load data
    df_rat = pd.read_json(args.rationales, lines=True)
    df_rew = pd.read_parquet(args.flow_rewards)
    
    df_samples = pd.read_parquet("data/labels/aligned_samples_h1.parquet")
    df_tech = pd.read_parquet("data/indicators/technical_event_tokens_h1.parquet")
    df_ctx = pd.merge(df_samples, df_tech[["sample_id", "technical_event_tokens", "regime_label"]], on="sample_id")
    
    # Merge rationales with rewards
    # Candidate rationales have sample_id, candidate_id, raw_text
    # Rewards have sample_id, candidate_id, flow_overall_reward
    df = pd.merge(df_rat, df_rew[['sample_id', 'candidate_id', 'flow_overall_reward']], on=['sample_id', 'candidate_id'], how='inner')
    
    # Ensure valid raw_text
    df = df.dropna(subset=['raw_text', 'flow_overall_reward'])
    
    preference_pairs = []
    
    # Group by sample_id
    grouped = df.groupby('sample_id')
    for sample_id, group in grouped:
        if len(group) < 2:
            continue
            
        group = group.sort_values('flow_overall_reward', ascending=False)
        
        row_ctx = df_ctx[df_ctx['sample_id'] == sample_id].iloc[0]
        ctx_vars = render_context(row_ctx)
        prompt_text = prompt_template
        for k, v in ctx_vars.items():
            prompt_text = prompt_text.replace(f"{{{k}}}", str(v))
            
        messages = [
            {"role": "system", "content": "You are a financial analyst."},
            {"role": "user", "content": prompt_text}
        ]
        prompt_formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        # Iterate over all possible pairs
        for i in range(len(group)):
            for j in range(i+1, len(group)):
                best = group.iloc[i]
                worst = group.iloc[j]
                
                margin = best['flow_overall_reward'] - worst['flow_overall_reward']
                if margin >= args.min_margin:
                    preference_pairs.append({
                        "sample_id": sample_id,
                        "prompt": prompt_formatted,
                        "chosen": best['raw_text'],
                        "rejected": worst['raw_text'],
                        "chosen_reward": float(best['flow_overall_reward']),
                        "rejected_reward": float(worst['flow_overall_reward']),
                        "margin": float(margin)
                    })
            
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for p in preference_pairs:
            f.write(json.dumps(p) + "\n")
            
    print(f"Created {len(preference_pairs)} preference pairs. Saved to {args.output}")

if __name__ == "__main__":
    main()
