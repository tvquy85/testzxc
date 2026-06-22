import json
import sys

import pandas as pd

from src.eval import repair_forecast_predictions_v6 as repair


def _raw(dist: dict[str, float], action: str = "hold") -> str:
    return json.dumps(
        {
            "forecast_distribution": {
                "Strong Down": dist["strong_down"],
                "Mild Down": dist["mild_down"],
                "Neutral": dist["neutral"],
                "Mild Up": dist["mild_up"],
                "Strong Up": dist["strong_up"],
            },
            "action": action,
        }
    )


def test_forecast_distribution_repair_normalizes_parse_ok_rows(tmp_path, monkeypatch):
    predictions = pd.DataFrame(
        [
            {
                "sample_id": "a",
                "split": "test",
                "pred_label": "invalid",
                "action": "invalid",
                "expected_action": "invalid",
                "action_consistency_ok": False,
                "p_strong_down": 0.0,
                "p_mild_down": 0.0,
                "p_neutral": 0.0,
                "p_mild_up": 0.0,
                "p_strong_up": 0.0,
                "parse_ok": True,
                "schema_ok": False,
                "parse_errors": ["forecast_distribution must sum to 1"],
                "raw_output": _raw(
                    {
                        "strong_down": 0.0,
                        "mild_down": 0.0,
                        "neutral": 0.20,
                        "mild_up": 0.0,
                        "strong_up": 0.30,
                    }
                ),
            },
            {
                "sample_id": "b",
                "split": "test",
                "pred_label": "neutral",
                "action": "hold",
                "expected_action": "hold",
                "action_consistency_ok": True,
                "p_strong_down": 0.0,
                "p_mild_down": 0.0,
                "p_neutral": 1.0,
                "p_mild_up": 0.0,
                "p_strong_up": 0.0,
                "parse_ok": True,
                "schema_ok": True,
                "parse_errors": [],
                "raw_output": _raw(
                    {
                        "strong_down": 0.0,
                        "mild_down": 0.0,
                        "neutral": 1.0,
                        "mild_up": 0.0,
                        "strong_up": 0.0,
                    }
                ),
            },
            {
                "sample_id": "c",
                "split": "test",
                "pred_label": "strong_down",
                "action": "short",
                "expected_action": "short",
                "action_consistency_ok": True,
                "p_strong_down": 1.0,
                "p_mild_down": 0.0,
                "p_neutral": 0.0,
                "p_mild_up": 0.0,
                "p_strong_up": 0.0,
                "parse_ok": True,
                "schema_ok": True,
                "parse_errors": [],
                "raw_output": _raw(
                    {
                        "strong_down": 1.0,
                        "mild_down": 0.0,
                        "neutral": 0.0,
                        "mild_up": 0.0,
                        "strong_up": 0.0,
                    },
                    action="short",
                ),
            },
        ]
    )
    contexts = pd.DataFrame(
        [
            {"sample_id": "a", "target_label_5": "strong_up"},
            {"sample_id": "b", "target_label_5": "neutral"},
            {"sample_id": "c", "target_label_5": "strong_down"},
        ]
    )
    predictions_path = tmp_path / "predictions.parquet"
    contexts_path = tmp_path / "contexts.parquet"
    output_path = tmp_path / "repaired.parquet"
    metrics_path = tmp_path / "metrics.json"
    status_path = tmp_path / "status.json"
    manifest_path = tmp_path / "manifest.json"
    predictions.to_parquet(predictions_path, index=False)
    contexts.to_parquet(contexts_path, index=False)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "repair_forecast_predictions_v6",
            "--predictions",
            str(predictions_path),
            "--contexts",
            str(contexts_path),
            "--output",
            str(output_path),
            "--metrics",
            str(metrics_path),
            "--status",
            str(status_path),
            "--manifest",
            str(manifest_path),
            "--min-schema-ok-rate",
            "1.0",
        ],
    )

    assert repair.main() == 0
    repaired = pd.read_parquet(output_path)
    repaired_a = repaired[repaired["sample_id"].eq("a")].iloc[0]
    assert bool(repaired_a["schema_ok"]) is True
    assert bool(repaired_a["forecast_repair_applied"]) is True
    assert abs(
        sum(float(repaired_a[f"p_{key}"]) for key in ["strong_down", "mild_down", "neutral", "mild_up", "strong_up"])
        - 1.0
    ) < 1e-9
    assert repaired_a["pred_label"] == "strong_up"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert metrics["repaired_rows"] == 1
    assert metrics["repaired_schema_ok_rate"] == 1.0
