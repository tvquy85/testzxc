from pathlib import Path

import pandas as pd


def test_baseline_table_has_required_methods():
    p = Path("outputs/tables/17_v6_comparable_baselines.csv")
    assert p.exists()
    methods = set(pd.read_csv(p)["method"])
    for name in ["Technical_Rule", "Qwen_DPO_V6", "Policy_Style_Scalar_Proxy"]:
        assert name in methods


def test_baseline_table_marks_reference_only_rows():
    p = Path("outputs/tables/17_v6_comparable_baselines.csv")
    assert p.exists()
    table = pd.read_csv(p)
    assert "reference_only" in table.columns
    pen = table[table["method"].eq("PEN_Reference_Only")]
    assert len(pen) == 1
    assert bool(pen.iloc[0]["reference_only"])
