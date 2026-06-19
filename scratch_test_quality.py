import pandas as pd

try:
    df = pd.read_json('data/rationales/candidate_rationales_h1.jsonl', lines=True)
    print(f"Total current rows: {len(df)}")
    if len(df) > 0:
        rate = df['schema_ok'].mean()
        print(f"Schema OK Rate: {rate:.2f}")
        
        if rate < 0.7:
            # Let's see what the reasons are
            failures = df[~df['schema_ok']].copy()
            print("\nExamples of failed raw_text:")
            for idx, row in failures.head(3).iterrows():
                print("-" * 50)
                print(row['raw_text'])
except Exception as e:
    print("Error:", e)
