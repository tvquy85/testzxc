import pandas as pd

try:
    df = pd.read_parquet('data/judge_outputs/flow_rewards_h1.parquet')
    print(f"Total Rows: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    if 'date' in df.columns:
        print(f"Date Range: {df['date'].min()} to {df['date'].max()}")
except Exception as e:
    print(f"Error reading fnspid_dataset.parquet: {e}")
    
try:
    df_labels = pd.read_parquet('data/processed/aligned_labels.parquet')
    print(f"Aligned Labels Rows: {len(df_labels)}")
    if 'date' in df_labels.columns:
        print(f"Labels Date Range: {df_labels['date'].min()} to {df_labels['date'].max()}")
except Exception as e:
    print(f"Error reading aligned_labels.parquet: {e}")
