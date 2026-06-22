import json
import sys

import pandas as pd

from src.eval import validation_calibrated_hybrid_v6 as hybrid


def _context(sample_id: str, label: str, token: str) -> dict:
    return {
        "sample_id": sample_id,
        "split": "val",
        "ticker": sample_id.upper(),
        "event_date": "2024-01-02",
        "target_label_5": label,
        "technical_event_tokens_json": json.dumps([{"direction_prior": token, "strength": "strong"}]),
    }


def _pred(sample_id: str, label: str, up: float, neutral: float = 0.4, schema_ok: bool = True) -> dict:
    return {
        "sample_id": sample_id,
        "schema_ok": schema_ok,
        "pred_label": label,
        "action": "long" if "up" in label else "hold",
        "p_strong_down": 0.0,
        "p_mild_down": 0.0,
        "p_neutral": neutral,
        "p_mild_up": up / 2.0,
        "p_strong_up": up / 2.0,
    }


def test_apply_gate_uses_dpo_only_when_schema_ok_and_confident():
    frame = pd.DataFrame(
        [
            {"schema_ok_bool": True, "dpo_confidence": 0.8, "dpo_pred": "strong_up", "technical_rule_pred": "neutral"},
            {"schema_ok_bool": False, "dpo_confidence": 0.9, "dpo_pred": "strong_up", "technical_rule_pred": "neutral"},
            {"schema_ok_bool": True, "dpo_confidence": 0.2, "dpo_pred": "strong_up", "technical_rule_pred": "neutral"},
        ]
    )
    assert hybrid.apply_gate(frame, 0.7).tolist() == ["strong_up", "neutral", "neutral"]


def test_validation_calibrated_hybrid_end_to_end(tmp_path, monkeypatch):
    val_contexts = pd.DataFrame(
        [
            _context("s1", "strong_up", "bearish_momentum"),
            _context("s2", "mild_down", "bearish_momentum"),
            _context("s3", "mild_down", "bearish_momentum"),
        ]
    )
    test_contexts = val_contexts.assign(split="test")
    val_predictions = pd.DataFrame(
        [
            _pred("s1", "strong_up", 0.9),
            _pred("s2", "strong_up", 0.2),
            _pred("s3", "strong_up", 0.2),
        ]
    )
    test_predictions = val_predictions.copy()

    val_contexts_path = tmp_path / "val_contexts.parquet"
    test_contexts_path = tmp_path / "test_contexts.parquet"
    val_predictions_path = tmp_path / "val_predictions.parquet"
    test_predictions_path = tmp_path / "test_predictions.parquet"
    output_path = tmp_path / "result.csv"
    threshold_path = tmp_path / "threshold.csv"
    pred_out_path = tmp_path / "hybrid.parquet"
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
            "validation_calibrated_hybrid_v6",
            "--val-contexts",
            str(val_contexts_path),
            "--val-predictions",
            str(val_predictions_path),
            "--test-contexts",
            str(test_contexts_path),
            "--test-predictions",
            str(test_predictions_path),
            "--threshold-grid",
            "0.00:1.00:0.10",
            "--output",
            str(output_path),
            "--threshold-table",
            str(threshold_path),
            "--predictions-output",
            str(pred_out_path),
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

    assert hybrid.main() == 0
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    pred_out = pd.read_parquet(pred_out_path)
    assert metrics["pipeline_pass"] is True
    assert metrics["best_threshold"] > 0.2
    assert pred_out["hybrid_pred"].tolist() == ["strong_up", "mild_down", "mild_down"]
