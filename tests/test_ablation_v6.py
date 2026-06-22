from pathlib import Path

import pandas as pd


def test_no_not_run_in_ablation():
    p = Path("outputs/tables/18_v6_ablation_results.csv")
    assert p.exists()
    df = pd.read_csv(p).astype(str)
    assert "NOT_RUN" not in set(df.values.ravel())


def test_required_v6_ablation_rows_present():
    p = Path("outputs/tables/18_v6_ablation_results.csv")
    assert p.exists()
    ablations = set(pd.read_csv(p)["ablation"])
    for name in ["Full_V6", "No_News_Evidence", "No_DPO_RWSFT_Only", "Technical_Rule"]:
        assert name in ablations
