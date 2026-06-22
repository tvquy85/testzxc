import json
import sys

import pandas as pd

from src.eval import validation_stacked_forecast_probe_v6 as stacker


def _context(sample_id: str, split: str, label: str, token_direction: str) -> dict:
    return {
        "sample_id": sample_id,
        "split": split,
        "ticker": sample_id.upper(),
        "event_date": "2024-01-02",
        "target_label_5": label,
        "technical_event_tokens_json": json.dumps([{"direction_prior": token_direction, "strength": "strong"}]),
        "num_company_event_evidence": 1,
        "num_context_only_evidence": 0,
        "mean_evidence_quality_score": 0.8,
        "num_hard_event_evidence": 1,
        "v6_training_weight": 1.0,
        "has_company_event_news": True,
        "has_hard_event_news": True,
        "no_news_context_flag": False,
        "v6_track": "hard_event_news",
        "evidence_pack_json": json.dumps({"company_evidence": [], "context_evidence": [], "technical_signals": []}),
    }


def _prediction(sample_id: str, pred_label: str, up: float) -> dict:
    return {
        "sample_id": sample_id,
        "schema_ok": True,
        "pred_label": pred_label,
        "p_strong_down": 0.0,
        "p_mild_down": 0.0,
        "p_neutral": 1.0 - up,
        "p_mild_up": up / 2.0,
        "p_strong_up": up / 2.0,
    }


def test_stacked_forecast_probe_end_to_end(tmp_path, monkeypatch):
    val_contexts = pd.DataFrame(
        [
            _context("s1", "val", "mild_up", "bullish_momentum"),
            _context("s2", "val", "mild_down", "bearish_momentum"),
            _context("s3", "val", "mild_up", "bullish_momentum"),
            _context("s4", "val", "mild_down", "bearish_momentum"),
            _context("s5", "val", "neutral", "neutral"),
        ]
    )
    test_contexts = val_contexts.assign(split="test")
    val_predictions = pd.DataFrame(
        [
            _prediction("s1", "mild_up", 0.8),
            _prediction("s2", "neutral", 0.1),
            _prediction("s3", "mild_up", 0.8),
            _prediction("s4", "neutral", 0.1),
            _prediction("s5", "neutral", 0.0),
        ]
    )
    test_predictions = val_predictions.copy()

    val_contexts_path = tmp_path / "val_contexts.parquet"
    test_contexts_path = tmp_path / "test_contexts.parquet"
    val_predictions_path = tmp_path / "val_predictions.parquet"
    test_predictions_path = tmp_path / "test_predictions.parquet"
    output_path = tmp_path / "result.csv"
    grid_path = tmp_path / "grid.csv"
    pred_path = tmp_path / "pred.parquet"
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
            "validation_stacked_forecast_probe_v6",
            "--val-contexts",
            str(val_contexts_path),
            "--val-predictions",
            str(val_predictions_path),
            "--test-contexts",
            str(test_contexts_path),
            "--test-predictions",
            str(test_predictions_path),
            "--c-grid",
            "0.1,1",
            "--output",
            str(output_path),
            "--grid-output",
            str(grid_path),
            "--predictions-output",
            str(pred_path),
            "--metrics",
            str(metrics_path),
            "--status",
            str(status_path),
            "--manifest",
            str(manifest_path),
            "--n-bootstrap",
            "5",
        ],
    )

    assert stacker.main() == 0
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert metrics["pipeline_pass"] is True
    assert metrics["claim_allowed"] is False
    assert metrics["feature_count"] > 0
    assert output_path.exists()
    assert grid_path.exists()
    assert pred_path.exists()
