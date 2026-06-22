import pandas as pd
from pathlib import Path

def test_v6_rationale_counts_and_schema():
    p=Path('data/rationales/parsed/current_v6_train_qwen3_1000x3.parquet')
    assert p.exists()
    df=pd.read_parquet(p)
    assert df['sample_id'].nunique()>=900
    assert len(df)>=2700
    assert df['schema_ok'].mean()>=0.95
