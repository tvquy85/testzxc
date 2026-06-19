import pandas as pd
import json
import yaml
import torch
import gc
import os
import argparse
from tqdm import tqdm

from technical_grounding_judge import score_technical_grounding
from utility_judge import score_utility
from news_nli_grounding_judge import NewsNLIGroundingJudge
from inferability_judge import build_inferability_prompts, parse_inferability_outputs
from financial_soundness_judge import build_financial_soundness_prompts, parse_financial_soundness_outputs

from transformers import AutoTokenizer, AutoModelForCausalLM

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rationales", required=True)
    parser.add_argument("--samples", required=True)
    parser.add_argument("--tech-features", required=False) # not strictly needed if we have tokens
    parser.add_argument("--tech-tokens", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        paths = yaml.safe_load(f)
        
    print("Loading data...")
    rationales = []
    with open(args.rationales, 'r', encoding='utf-8') as f:
        for line in f:
            rationales.append(json.loads(line))
            if args.limit and len(rationales) >= args.limit:
                break
    df = pd.DataFrame(rationales)
    print(f"Loaded {len(df)} rationales.")

    df_samples = pd.read_parquet(args.samples)
    df_tech = pd.read_parquet(args.tech_tokens)

    df = df.merge(df_samples[['sample_id', 'headline', 'abnormal_return_h1', 'label_5']], on='sample_id', how='left')
    df = df.merge(df_tech[['sample_id', 'technical_event_tokens']], on='sample_id', how='left')

    headlines = df['headline'].fillna("").tolist()
    rationales_text = df['raw_text'].fillna("").tolist()
    tech_tokens = df['technical_event_tokens'].fillna("").tolist()
    abnormal_returns = df['abnormal_return_h1'].fillna(0.0).tolist()
    
    actions = []
    for r in df['rationale_json']:
        if isinstance(r, dict) and 'action' in r:
            actions.append(r['action'])
        else:
            actions.append('hold')

    print("1. Running News NLI Grounding Judge...")
    nli_path = paths['models']['nli_cross_encoder']
    nli_judge = NewsNLIGroundingJudge(nli_path)
    nli_results = nli_judge.score_batch(headlines, rationales_text, batch_size=32)
    
    df['news_entailment_rate'] = [res['entailment'] for res in nli_results]
    df['news_contradiction_rate'] = [res['contradiction'] for res in nli_results]

    # Clear NLI from GPU
    del nli_judge
    torch.cuda.empty_cache()
    gc.collect()

    print("2. Running Technical Grounding Judge & Utility Judge...")
    tech_grounding_scores = []
    utility_scores = []
    for rt, tt, ac, ar in zip(rationales_text, tech_tokens, actions, abnormal_returns):
        tech_grounding_scores.append(score_technical_grounding(rt, tt))
        utility_scores.append(score_utility(ac, ar))
        
    df['technical_grounding_score'] = tech_grounding_scores
    df['utility_score'] = utility_scores

    print("3. Running LLM Judges (Inferability & Financial Soundness)...")
    llm_path = paths['models']['llama3_judge']
    print(f"Loading HF pipeline for {llm_path}")
    
    # RTX 3090 has 24GB. 4B model fits well with max_model_len 4096.
    tokenizer = AutoTokenizer.from_pretrained(llm_path)
    # Some older Qwen tokenizer versions might not have a default chat template or padding side
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = 'left'

    # Add kv_cache_dtype for memory optimization if needed, but bfloat16 is fine.
    model = AutoModelForCausalLM.from_pretrained(llm_path, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True)
    
    print("   -> Inferability Judge")
    infer_prompts = build_inferability_prompts(headlines, tech_tokens, rationales_text, tokenizer)
    infer_texts = []
    batch_size = 4
    for i in tqdm(range(0, len(infer_prompts), batch_size), desc="Inferability"):
        batch = infer_prompts[i:i+batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=2048).to(model.device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=256, temperature=0.0, do_sample=False)
        for j in range(len(batch)):
            generated_ids = outputs[j][inputs.input_ids[j].shape[-1]:]
            infer_texts.append(tokenizer.decode(generated_ids, skip_special_tokens=True))
            
    infer_results = parse_inferability_outputs(infer_texts)

    infer_p_strong_down = []
    infer_p_mild_down = []
    infer_p_neutral = []
    infer_p_mild_up = []
    infer_p_strong_up = []
    infer_pred_label = []
    infer_prob_true_label = []

    for idx, (res, true_lbl) in enumerate(zip(infer_results, df['label_5'].tolist())):
        lbl_map = {
            'Strong Down': 'strong_down',
            'Mild Down': 'mild_down',
            'Neutral': 'neutral',
            'Mild Up': 'mild_up',
            'Strong Up': 'strong_up'
        }
        mapped_true_lbl = lbl_map.get(true_lbl, 'neutral')
        
        infer_p_strong_down.append(res.get('strong_down', 0.0))
        infer_p_mild_down.append(res.get('mild_down', 0.0))
        infer_p_neutral.append(res.get('neutral', 0.0))
        infer_p_mild_up.append(res.get('mild_up', 0.0))
        infer_p_strong_up.append(res.get('strong_up', 0.0))
        
        # Get max prob label
        pred_lbl = max(res, key=res.get) if res else 'neutral'
        infer_pred_label.append(pred_lbl)
        
        # Get prob of true label
        prob_true = res.get(mapped_true_lbl, 0.0)
        infer_prob_true_label.append(prob_true)

    df['infer_p_strong_down'] = infer_p_strong_down
    df['infer_p_mild_down'] = infer_p_mild_down
    df['infer_p_neutral'] = infer_p_neutral
    df['infer_p_mild_up'] = infer_p_mild_up
    df['infer_p_strong_up'] = infer_p_strong_up
    df['infer_pred_label'] = infer_pred_label
    df['infer_prob_true_label'] = infer_prob_true_label

    print("   -> Financial Soundness Judge")
    fs_prompts = build_financial_soundness_prompts(headlines, tech_tokens, rationales_text, tokenizer)
    fs_texts = []
    for i in tqdm(range(0, len(fs_prompts), batch_size), desc="Financial Soundness"):
        batch = fs_prompts[i:i+batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=2048).to(model.device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=256, temperature=0.0, do_sample=False)
        for j in range(len(batch)):
            generated_ids = outputs[j][inputs.input_ids[j].shape[-1]:]
            fs_texts.append(tokenizer.decode(generated_ids, skip_special_tokens=True))
            
    fs_results = parse_financial_soundness_outputs(fs_texts)

    df['financial_soundness_score'] = [r['financial_soundness'] for r in fs_results]
    df['overconfidence_score'] = [r['overconfidence'] for r in fs_results]
    # groundedness from fs_results is also computed but formula doesn't strictly use it, we keep it just in case
    df['financial_groundedness'] = [r['groundedness'] for r in fs_results]

    del model
    torch.cuda.empty_cache()
    gc.collect()

    print("4. Computing overall proxy score...")
    overall_scores = (
        0.40 * df['infer_prob_true_label'] +
        0.20 * df['technical_grounding_score'] +
        0.15 * df['news_entailment_rate'] -
        0.15 * df['news_contradiction_rate'] +
        0.10 * df['utility_score'] +
        0.10 * df['financial_soundness_score']
    )
    df['overall_proxy_score'] = overall_scores.clip(lower=0.0, upper=1.0)

    # Required columns format
    cols_to_keep = [
        'sample_id', 'candidate_id', 'label_5',
        'infer_p_strong_down', 'infer_p_mild_down', 'infer_p_neutral', 'infer_p_mild_up', 'infer_p_strong_up',
        'infer_pred_label', 'infer_prob_true_label',
        'financial_soundness_score', 'overconfidence_score',
        'technical_grounding_score', 'news_entailment_rate', 'news_contradiction_rate',
        'utility_score', 'overall_proxy_score'
    ]
    df_out = df[cols_to_keep].copy()

    # Create dir if not exist
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df_out.to_parquet(args.output, index=False)
    print(f"Saved to {args.output}")

    # Compute status checks
    valid_mask = df_out['overall_proxy_score'].notna() & df_out['overall_proxy_score'].between(0.0, 1.0)
    valid_rate = valid_mask.mean()
    unique_samples = df_out['sample_id'].nunique()
    mean_score = df_out['overall_proxy_score'].mean()
    passed = valid_rate >= 0.95 and unique_samples >= 400

    status = {
        "step": "09_PROXY_JUDGES_AND_GROUNDING",
        "status": "PASS" if passed else "FAIL",
        "scored_rows": len(df_out),
        "unique_samples": unique_samples,
        "valid_score_rate": float(valid_rate),
        "mean_overall_proxy_score": float(mean_score),
        "models_used": [nli_path, llm_path],
        "notes": "Judges applied."
    }

    os.makedirs("outputs/status", exist_ok=True)
    with open("outputs/status/09_PROXY_JUDGES_AND_GROUNDING.status.json", "w") as f:
        json.dump(status, f, indent=2)

    print(json.dumps(status, indent=2))

if __name__ == "__main__":
    main()
