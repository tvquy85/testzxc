from pathlib import Path

import pandas as pd


def test_v6_prediction_files():
    p = Path("outputs/predictions/current_v6_dpo_predictions.parquet")
    assert p.exists()
    df = pd.read_parquet(p)
    assert len(df) >= 300
    assert df["schema_ok"].mean() >= 0.90


def test_v6_prediction_contexts_are_test_only():
    p = Path("data/processed/current_v6_prediction_contexts.parquet")
    assert p.exists()
    df = pd.read_parquet(p)
    assert len(df) >= 300
    assert set(df["split"].dropna()) == {"test"}
