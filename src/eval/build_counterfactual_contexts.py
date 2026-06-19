import pandas as pd
import json
import argparse
import os

def flip_regime(tokens_str):
    if "[HIGH_VOLATILITY_REGIME" in tokens_str:
        return tokens_str.replace("[HIGH_VOLATILITY_REGIME", "[LOW_VOLATILITY_REGIME")
    elif "[LOW_VOLATILITY_REGIME" in tokens_str:
        return tokens_str.replace("[LOW_VOLATILITY_REGIME", "[HIGH_VOLATILITY_REGIME")
    # if no vol regime is explicitly found, just append it
    return tokens_str + " [HIGH_VOLATILITY_REGIME: vol20=9.99, strength=strong]"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    print("Loading test set...")
    df_samples = pd.read_parquet("data/labels/aligned_samples_h1.parquet")
    df_tech = pd.read_parquet("data/indicators/technical_event_tokens_h1.parquet")
    
    df = df_samples.merge(df_tech[['sample_id', 'technical_event_tokens']], on='sample_id', how='left')
    df['window_end_date'] = pd.to_datetime(df['window_end_date'])
    df = df.sort_values('window_end_date')
    
    test_size = int(len(df) * 0.15)
    df_test = df.tail(test_size).copy()
    
    if args.limit:
        df_test = df_test.tail(args.limit).copy()

    with open("prompts/rationale_generation_prompt.txt", "r", encoding="utf-8") as f:
        prompt_template = f.read()

    cf_data = []
    
    for row in df_test.itertuples():
        hl = row.headline if pd.notna(row.headline) else ""
        tt = row.technical_event_tokens if pd.notna(row.technical_event_tokens) else ""
        
        # Original
        p_orig = prompt_template.replace("{{headline}}", hl).replace("{{technical_event_tokens}}", tt)
        cf_data.append({
            "sample_id": row.sample_id,
            "cf_type": "ORIGINAL",
            "prompt": p_orig
        })
        
        # 1. CF_NEWS_REMOVE
        p_no_news = prompt_template.replace("{{headline}}", "").replace("{{technical_event_tokens}}", tt)
        cf_data.append({
            "sample_id": row.sample_id,
            "cf_type": "CF_NEWS_REMOVE",
            "prompt": p_no_news
        })
        
        # 2. CF_TECH_NEUTRALIZE
        p_no_tech = prompt_template.replace("{{headline}}", hl).replace("{{technical_event_tokens}}", "[TECH_NEUTRAL: All indicators are neutral and uninformative]")
        cf_data.append({
            "sample_id": row.sample_id,
            "cf_type": "CF_TECH_NEUTRALIZE",
            "prompt": p_no_tech
        })
        
        # 3. CF_REGIME_FLIP
        flipped_tt = flip_regime(tt)
        p_flip = prompt_template.replace("{{headline}}", hl).replace("{{technical_event_tokens}}", flipped_tt)
        cf_data.append({
            "sample_id": row.sample_id,
            "cf_type": "CF_REGIME_FLIP",
            "prompt": p_flip
        })

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for item in cf_data:
            f.write(json.dumps(item) + "\n")
            
    print(f"Generated {len(cf_data)} counterfactual prompts to {args.output}")

if __name__ == "__main__":
    main()
