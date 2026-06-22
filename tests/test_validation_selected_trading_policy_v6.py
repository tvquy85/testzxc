import json
import sys

import pandas as pd

from src.eval import validation_selected_trading_policy_v6 as policy


def _contexts(split: str) -> pd.DataFrame:
    rows = []
    for idx, ret in enumerate([0.02, -0.03, 0.015, -0.02], start=1):
        rows.append(
            {
                "sample_id": f"{split}{idx}",
                "ticker": f"T{idx}",
                "event_date": f"2024-01-{idx + 1:02d}",
                "split": split,
                "target_return": ret,
                "technical_event_tokens_json": json.dumps([{"direction_prior": "bullish", "strength": "strong"}]),
            }
        )
    return pd.DataFrame(rows)


def _predictions(split: str) -> pd.DataFrame:
    rows = []
    for idx, confidence in enumerate([0.9, 0.4, 0.9, 0.4], start=1):
        rows.append(
            {
                "sample_id": f"{split}{idx}",
                "split": split,
                "schema_ok": True,
                "action": "long",
                "pred_label": "mild_up",
                "p_strong_down": 0.02,
                "p_mild_down": 0.03,
                "p_neutral": 0.05,
                "p_mild_up": confidence,
                "p_strong_up": 0.0,
            }
        )
    return pd.DataFrame(rows)


def test_validation_selected_policy_uses_validation_grid(tmp_path, monkeypatch):
    val_contexts = _contexts("val")
    test_contexts = _contexts("test")
    val_predictions = _predictions("val")
    test_predictions = _predictions("test")
    val_contexts_path = tmp_path / "val_contexts.parquet"
    test_contexts_path = tmp_path / "test_contexts.parquet"
    val_predictions_path = tmp_path / "val_predictions.parquet"
    test_predictions_path = tmp_path / "test_predictions.parquet"
    grid_path = tmp_path / "grid.csv"
    summary_path = tmp_path / "summary.csv"
    daily_path = tmp_path / "daily.csv"
    metrics_path = tmp_path / "metrics.json"
    status_path = tmp_path / "status.json"
    manifest_path = tmp_path / "manifest.json"
    val_contexts.to_parquet(val_contexts_path)
    test_contexts.to_parquet(test_contexts_path)
    val_predictions.to_parquet(val_predictions_path)
    test_predictions.to_parquet(test_predictions_path)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "validation_selected_trading_policy_v6",
            "--val-contexts",
            str(val_contexts_path),
            "--val-predictions",
            str(val_predictions_path),
            "--test-contexts",
            str(test_contexts_path),
            "--test-predictions",
            str(test_predictions_path),
            "--thresholds",
            "0.00,0.80",
            "--position-caps",
            "1,2",
            "--min-val-nonzero-days",
            "1",
            "--grid-output",
            str(grid_path),
            "--summary-output",
            str(summary_path),
            "--daily-output",
            str(daily_path),
            "--metrics",
            str(metrics_path),
            "--status",
            str(status_path),
            "--manifest",
            str(manifest_path),
            "--n-bootstrap",
            "5",
            "--block-size",
            "1",
        ],
    )

    assert policy.main() == 0
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert metrics["pipeline_pass"] is True
    assert metrics["selection_uses_test_returns"] is False
    assert metrics["selected_threshold"] == 0.8
    assert metrics["claim_allowed"] is False
    assert metrics["alpha_paper_level_supported"] is False
    assert grid_path.exists()
    assert summary_path.exists()
    assert daily_path.exists()
