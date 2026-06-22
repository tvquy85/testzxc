import json
import sys

import pandas as pd

from src.eval import audit_counterfactual_eligibility_v6 as audit


def test_side_signal_eligible_uses_expected_side_mass():
    up_row = pd.Series({"schema_ok_bool": True, "expected_side": "up", "up_side": 0.30, "down_side": 0.00})
    down_row = pd.Series({"schema_ok_bool": True, "expected_side": "down", "up_side": 0.80, "down_side": 0.05})
    assert audit.side_signal_eligible(up_row, up_threshold=0.25, down_threshold=0.20)
    assert not audit.side_signal_eligible(down_row, up_threshold=0.25, down_threshold=0.20)


def test_counterfactual_eligibility_end_to_end(tmp_path, monkeypatch):
    tasks = [
        {
            "sample_id": "s1",
            "counterfactual_type": "remove_positive_evidence",
            "expected_direction": "up_decreases",
        },
        {
            "sample_id": "s2",
            "counterfactual_type": "remove_negative_evidence",
            "expected_direction": "down_decreases",
        },
    ]
    tasks_path = tmp_path / "tasks.jsonl"
    tasks_path.write_text("\n".join(json.dumps(row) for row in tasks) + "\n", encoding="utf-8")
    preds = pd.DataFrame(
        [
            {
                "sample_id": "s1",
                "schema_ok": True,
                "pred_label": "mild_up",
                "action": "long",
                "p_strong_down": 0.0,
                "p_mild_down": 0.0,
                "p_neutral": 0.4,
                "p_mild_up": 0.3,
                "p_strong_up": 0.3,
            },
            {
                "sample_id": "s2",
                "schema_ok": True,
                "pred_label": "neutral",
                "action": "hold",
                "p_strong_down": 0.0,
                "p_mild_down": 0.05,
                "p_neutral": 0.2,
                "p_mild_up": 0.4,
                "p_strong_up": 0.35,
            },
        ]
    )
    preds_path = tmp_path / "preds.parquet"
    breakdown_path = tmp_path / "breakdown.csv"
    output_path = tmp_path / "by_type.csv"
    task_output_path = tmp_path / "task_out.csv"
    metrics_path = tmp_path / "metrics.json"
    status_path = tmp_path / "status.json"
    manifest_path = tmp_path / "manifest.json"
    preds.to_parquet(preds_path)
    pd.DataFrame(
        [
            {"counterfactual_type": "remove_positive_evidence", "pass_rate": 0.5, "no_change_rate": 0.2, "wrong_direction_rate": 0.3},
            {"counterfactual_type": "remove_negative_evidence", "pass_rate": 0.1, "no_change_rate": 0.8, "wrong_direction_rate": 0.1},
        ]
    ).to_csv(breakdown_path, index=False)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "audit_counterfactual_eligibility_v6",
            "--tasks",
            str(tasks_path),
            "--predictions",
            str(preds_path),
            "--breakdown",
            str(breakdown_path),
            "--output",
            str(output_path),
            "--task-output",
            str(task_output_path),
            "--metrics",
            str(metrics_path),
            "--status",
            str(status_path),
            "--manifest",
            str(manifest_path),
        ],
    )

    assert audit.main() == 0
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    task_table = pd.read_csv(task_output_path)
    assert metrics["tasks"] == 2
    assert metrics["eligible_side_signal_rate"] == 0.5
    assert task_table["eligible_side_signal"].tolist() == [True, False]
