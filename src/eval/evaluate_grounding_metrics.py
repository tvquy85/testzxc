import pandas as pd
import json
import argparse
import os
import yaml
import torch
import gc

import sys
sys.path.append('src/judges')
from news_nli_grounding_judge import NewsNLIGroundingJudge

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    with open("configs/local_paths.yaml", 'r') as f:
        paths = yaml.safe_load(f)

    data = []
    with open(args.pred, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
            
    df = pd.DataFrame(data)
    print(f"Loaded {len(df)} predictions for grounding evaluation.")

    headlines = df['headline'].fillna("").tolist()
    rationales_text = df['rationale'].fillna("").tolist()

    print("Running News NLI Grounding Judge...")
    nli_path = paths['models']['nli_cross_encoder']
    nli_judge = NewsNLIGroundingJudge(nli_path)
    nli_results = nli_judge.score_batch(headlines, rationales_text, batch_size=32)
    
    df['news_entailment_rate'] = [res['entailment'] for res in nli_results]
    df['news_contradiction_rate'] = [res['contradiction'] for res in nli_results]

    avg_entailment = df['news_entailment_rate'].mean()
    avg_contradiction = df['news_contradiction_rate'].mean()

    metrics = {
        "mean_entailment": float(avg_entailment),
        "mean_contradiction": float(avg_contradiction),
        "hallucination_rate": float((df['news_contradiction_rate'] > 0.5).mean())
    }
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
        
    print(json.dumps(metrics, indent=2))
    print(f"Grounding metrics saved to {args.output}")

if __name__ == "__main__":
    main()
