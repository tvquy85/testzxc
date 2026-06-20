import os
import json
import pandas as pd

os.makedirs('outputs/data_samples', exist_ok=True)

# 1. Sample DPO pairs
try:
    with open('data/alignment/dpo_pairs_train_v2.jsonl', 'r', encoding='utf-8') as f, open('outputs/data_samples/sample_dpo_pairs.jsonl', 'w', encoding='utf-8') as out:
        for _ in range(5):
            out.write(f.readline())
except Exception as e:
    print("Error sampling DPO:", e)

# 2. Sample raw rationale
try:
    with open('data/rationales/raw/train_qwen3_4b_stage1_bulk.jsonl', 'r', encoding='utf-8') as f, open('outputs/data_samples/sample_raw_rationale.jsonl', 'w', encoding='utf-8') as out:
        for _ in range(5):
            out.write(f.readline())
except Exception as e:
    print("Error sampling raw rationale:", e)

# 3. Sample parsed rationale
try:
    df = pd.read_parquet('data/rationales/parsed/train_candidates_stage1_strict.parquet')
    df.head(5).to_json('outputs/data_samples/sample_parsed_rationale.json', orient='records', lines=True, force_ascii=False)
except Exception as e:
    print("Error sampling parsed rationale:", e)

# 4. Sample judge
try:
    df = pd.read_parquet('data/judges/inferability_multi_judge_stage1.parquet')
    df.head(5).to_json('outputs/data_samples/sample_judge_inferability.json', orient='records', lines=True, force_ascii=False)
except Exception as e:
    print("Error sampling judge:", e)

# 5. Sample grounding
try:
    df = pd.read_parquet('data/judges/claim_grounding_scores_stage1.parquet')
    df.head(10).to_json('outputs/data_samples/sample_claim_grounding.json', orient='records', lines=True, force_ascii=False)
except Exception as e:
    print("Error sampling grounding:", e)

# 6. Sample technical tokens
try:
    df = pd.read_parquet('data/indicators/technical_event_tokens_h1_v2.parquet')
    df.head(5).to_json('outputs/data_samples/sample_technical_tokens.json', orient='records', lines=True, force_ascii=False)
except Exception as e:
    print("Error sampling technical tokens:", e)

print("Samples created successfully in outputs/data_samples/")
