import json

import pandas as pd

from src.eval.build_counterfactual_evidence_v6 import TASK_TYPES, build_task
from src.eval.counterfactual_direction_rules_v6 import expected_direction, normalized_expected_direction


def test_expected_direction_positive_removed():
    assert expected_direction("remove_positive_evidence") == "up_decreases"


def test_expected_direction_negative_removed():
    assert expected_direction("remove_negative_evidence") == "down_decreases"


def test_expected_direction_aliases_match_evaluator_contract():
    assert normalized_expected_direction("up_decreases") == "up_decrease"
    assert normalized_expected_direction("down_decreases") == "down_decrease"


def test_build_counterfactual_evidence_v6_all_task_types():
    row = pd.Series(
        {
            "sample_id": "s1",
            "ticker": "AAA",
            "event_date": "2024-01-02",
            "split": "test",
            "v6_track": "hard_event_news",
            "evidence_pack_json": json.dumps(
                {
                    "company_evidence": [
                        {
                            "headline": "AAA reports strong profit growth but warns of weak demand",
                            "body_excerpt": "The company beat expectations, but losses and lower guidance remain risks.",
                            "evidence_id": "N1",
                        },
                        {
                            "headline": "Analysts upgrade AAA after record growth",
                            "body_excerpt": "The upgrade reflects strong execution and higher profit expectations.",
                            "evidence_id": "N2",
                        }
                    ],
                    "context_evidence": [],
                    "technical_signals": [
                        {"token": "MACD_BEARISH", "direction_prior": "bearish_momentum", "rule": "MACD_hist < 0"},
                        {"token": "PRICE_ABOVE_SMA20", "direction_prior": "bullish_trend", "rule": "price_vs_SMA20 > 0"},
                    ],
                }
            ),
        }
    )

    built = {cf_type: build_task(row, cf_type) for cf_type in TASK_TYPES}
    assert all(task is not None for task in built.values())
    assert built["remove_positive_evidence"]["expected_direction"] == "up_decreases"
    assert built["remove_negative_evidence"]["expected_direction"] == "down_decreases"
    assert "strong" not in built["neutralize_positive_evidence"]["counterfactual_body"].lower()
    assert "weak" not in built["neutralize_negative_evidence"]["counterfactual_headline"].lower()
    assert "MACD_BEARISH" not in built["neutralize_bearish_technical"]["counterfactual_technical_event_tokens_json"]
    assert "PRICE_ABOVE_SMA20" not in built["neutralize_bullish_technical"]["counterfactual_technical_event_tokens_json"]
