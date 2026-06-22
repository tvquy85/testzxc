from types import SimpleNamespace

import pandas as pd

from src.eval.generate_test_predictions_v2 import prediction_quality_failures


def test_prediction_quality_fails_below_min_rows():
    out = pd.DataFrame(
        [
            {"split": "test", "parse_ok": True, "schema_ok": True, "action": "long"},
            {"split": "test", "parse_ok": True, "schema_ok": True, "action": "short"},
        ]
    )
    args = SimpleNamespace(
        min_rows=3,
        min_parse_ok_rate=0.8,
        min_schema_ok_rate=0.8,
        allow_fallback=False,
        allow_non_dpo_checkpoint=True,
        min_trading_days=0,
        split="test",
        allow_non_test_split=False,
    )

    failures = prediction_quality_failures(out, args, checkpoint_used="outputs/models/dpo", checkpoint_source="primary")

    assert any("prediction rows 2 < 3" in failure for failure in failures)


def test_prediction_quality_allows_explicit_validation_split():
    out = pd.DataFrame(
        [
            {"split": "val", "parse_ok": True, "schema_ok": True, "action": "long"},
            {"split": "val", "parse_ok": True, "schema_ok": True, "action": "short"},
        ]
    )
    args = SimpleNamespace(
        min_rows=2,
        min_parse_ok_rate=0.8,
        min_schema_ok_rate=0.8,
        allow_fallback=False,
        allow_non_dpo_checkpoint=True,
        min_trading_days=0,
        split="val",
        allow_non_test_split=True,
    )

    failures = prediction_quality_failures(out, args, checkpoint_used="outputs/models/dpo", checkpoint_source="primary")

    assert not failures
