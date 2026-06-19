import pandas as pd

from src.alignment.build_dpo_pairs_v2 import build_dpo_pairs
from src.data.build_train_conflict_subset_v2 import build_subset


def test_conflict_subset_is_train_only_and_prioritizes_hard_rows():
    samples = pd.DataFrame(
        [
            {"sample_id": "a", "event_date": "2020-01-01", "headline": "earnings beat", "body": "", "label_5": "strong_up"},
            {"sample_id": "b", "event_date": "2020-01-02", "headline": "neutral item", "body": "", "label_5": "neutral"},
            {"sample_id": "c", "event_date": "2022-01-01", "headline": "lawsuit warning", "body": "", "label_5": "strong_down"},
        ]
    )
    splits = pd.DataFrame(
        [
            {"sample_id": "a", "split": "train"},
            {"sample_id": "b", "split": "train"},
            {"sample_id": "c", "split": "val"},
        ]
    )
    tokens = pd.DataFrame(
        [
            {
                "sample_id": "a",
                "regime_label": "high_vol",
                "technical_event_tokens_json": '[{"token":"MACD_BEARISH","direction_prior":"bearish","strength":"high"}]',
            },
            {
                "sample_id": "b",
                "regime_label": "normal_vol",
                "technical_event_tokens_json": '[{"token":"MACD_BULLISH","direction_prior":"bullish","strength":"medium"}]',
            },
            {
                "sample_id": "c",
                "regime_label": "high_vol",
                "technical_event_tokens_json": '[{"token":"MACD_BEARISH","direction_prior":"bearish","strength":"high"}]',
            },
        ]
    )

    subset = build_subset(samples, splits, tokens, limit=2, seed=1)

    assert set(subset["split"]) == {"train"}
    assert "c" not in set(subset["sample_id"])
    assert subset.iloc[0]["sample_id"] == "a"


def test_dpo_pairs_are_same_sample_and_ordered_by_score_gap():
    scored = pd.DataFrame(
        [
            {"sample_id": "s1", "candidate_id": 0, "split": "train", "raw_text": "bad", "alignment_proxy_score": 0.1},
            {"sample_id": "s1", "candidate_id": 1, "split": "train", "raw_text": "ok", "alignment_proxy_score": 0.4},
            {"sample_id": "s1", "candidate_id": 2, "split": "train", "raw_text": "best", "alignment_proxy_score": 0.9},
            {"sample_id": "s2", "candidate_id": 0, "split": "val", "raw_text": "no", "alignment_proxy_score": 0.0},
            {"sample_id": "s2", "candidate_id": 1, "split": "val", "raw_text": "no", "alignment_proxy_score": 1.0},
        ]
    )

    pairs = build_dpo_pairs(scored, min_gap=0.2, max_pairs_per_sample=3)

    assert pairs
    assert {pair["sample_id"] for pair in pairs} == {"s1"}
    assert all(pair["split"] == "train" for pair in pairs)
    assert all(pair["chosen_score"] > pair["rejected_score"] for pair in pairs)
    assert all(pair["chosen"] != pair["rejected"] for pair in pairs)
