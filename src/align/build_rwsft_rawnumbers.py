import pandas as pd
import json
import re
import os

def main():
    print("Loading datasets...")
    df_train = pd.read_json("data/rationales/rwsft_train_h1.jsonl", lines=True)
    df_features = pd.read_parquet("data/indicators/technical_features_h1.parquet")
    
    ignore_cols = ['sample_id', 'ticker', 'date', 'window_start_date', 'window_end_date']
    feature_cols = [c for c in df_features.columns if c not in ignore_cols and pd.api.types.is_numeric_dtype(df_features[c])]
    
    print("Formatting raw features...")
    features_dict = {}
    for row in df_features.itertuples():
        vals = [f"{c}: {getattr(row, c):.4f}" for c in feature_cols if pd.notnull(getattr(row, c))]
        features_dict[row.sample_id] = ", ".join(vals)
        
    def replace_tokens(row):
        msgs = row['messages']
        sample_id = row['sample_id']
        raw_feats = features_dict.get(sample_id, "No technical features")
        
        new_msgs = []
        for m in msgs:
            if m['role'] == 'user':
                # Replace Technical Indicator Tokens block with Raw Numbers
                content = m['content']
                # The block looks like:
                # Technical Indicator Tokens:
                # [MACD_BEARISH: hist=-0.98]
                # [PRICE_BELOW_SMA20: distance=-0.3%]
                # 
                # Instructions:
                # We want to replace everything between "Technical Indicator Tokens:\n" and "\n\nInstructions:"
                # Using regex
                pattern = r"Technical Indicator Tokens:\n(.*?)\n\nInstructions:"
                replacement = f"Technical Indicator Tokens:\n{raw_feats}\n\nInstructions:"
                new_content = re.sub(pattern, replacement, content, flags=re.DOTALL)
                new_msgs.append({'role': m['role'], 'content': new_content})
            else:
                new_msgs.append(m)
        return new_msgs

    print("Replacing tokens with raw numbers in train set...")
    df_train['messages'] = df_train.apply(replace_tokens, axis=1)
    
    output_path = "data/rationales/rwsft_train_rawnumbers.jsonl"
    with open(output_path, "w", encoding="utf-8") as f:
        for idx, row in df_train.iterrows():
            f.write(json.dumps({
                "sample_id": row["sample_id"],
                "messages": row["messages"],
                "weight": row["weight"],
                "flow_reward": row["flow_reward"]
            }) + "\n")
            
    print(f"Saved {len(df_train)} samples to {output_path}")

if __name__ == "__main__":
    main()
