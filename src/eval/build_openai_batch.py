import json
import pandas as pd
from src.llm.render_context import render_context

def main():
    print("Loading test dataset...")
    df_samples = pd.read_parquet("data/labels/aligned_samples_h1.parquet")
    df_samples['timestamp_utc'] = pd.to_datetime(df_samples['timestamp_utc'])
    df_samples = df_samples.sort_values("timestamp_utc")
    
    cutoff_idx = int(len(df_samples) * 0.85)
    test_df = df_samples.iloc[cutoff_idx:].copy()
    
    df_tech = pd.read_parquet("data/indicators/technical_event_tokens_h1.parquet")
    test_df = pd.merge(test_df, df_tech[['sample_id', 'technical_event_tokens', 'regime_label']], on='sample_id', how='inner')
    
    with open("prompts/rationale_generation_prompt.txt", "r", encoding="utf-8") as f:
        prompt_template = f.read()
        
    print(f"Building {len(test_df)} requests for OpenAI Batch API...")
    batch_requests = []
    
    for idx, row in test_df.iterrows():
        ctx = render_context(row)
        prompt_text = prompt_template
        for k, v in ctx.items():
            prompt_text = prompt_text.replace(f"{{{k}}}", str(v))
            
        request = {
            "custom_id": row['sample_id'],
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": "gpt-4o-mini", # Or gpt-4o
                "messages": [
                    {"role": "system", "content": "You are an expert financial analyst."},
                    {"role": "user", "content": prompt_text}
                ],
                "max_tokens": 500,
                "temperature": 0.2
            }
        }
        batch_requests.append(request)
        
    out_dir = "outputs/openai_batches"
    os.makedirs(out_dir, exist_ok=True)
    
    # Split into chunks to avoid the 2,000,000 enqueued token limit
    # Average tokens per request = ~1000 prompt + 500 max_tokens = 1500 tokens
    # 2,000,000 / 1500 ≈ 1333. Let's use chunks of 1000 requests to be safe.
    chunk_size = 1000
    
    for i in range(0, len(batch_requests), chunk_size):
        chunk = batch_requests[i:i + chunk_size]
        part_idx = i // chunk_size + 1
        out_path = f"{out_dir}/openai_batch_input_part{part_idx}.jsonl"
        
        with open(out_path, "w", encoding="utf-8") as f:
            for req in chunk:
                f.write(json.dumps(req) + "\n")
                
    print(f"Done! Saved {len(batch_requests)} requests across {len(range(0, len(batch_requests), chunk_size))} files in {out_dir}/")
    print("Upload them sequentially to https://platform.openai.com/batches")

if __name__ == "__main__":
    import os
    main()
