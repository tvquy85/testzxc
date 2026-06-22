import json
import sys

import pandas as pd

from src.eval import repaired_forecast_baseline_probe_v6 as probe


def _pred(sample_id: str, label: str) -> dict:
    return {
        "sample_id": sample_id,
        "split": "test",
        "pred_label": label,
        "action": "hold",
        "schema_ok": True,
    }


def test_repaired_forecast_baseline_probe_reports_alignment_signal(tmp_path, monkeypatch):
    contexts = pd.DataFrame(
        [
            {"sample_id": "a", "split": "test", "target_label_5": "strong_up", "evidence_pack_json": "{}"},
            {"sample_id": "b", "split": "test", "target_label_5": "neutral", "evidence_pack_json": "{}"},
            {"sample_id": "c", "split": "test", "target_label_5": "mild_down", "evidence_pack_json": "{}"},
        ]
    )
    dpo_original = pd.DataFrame([_pred("a", "neutral"), _pred("b", "neutral"), _pred("c", "neutral")])
    dpo_repaired = pd.DataFrame([_pred("a", "strong_up"), _pred("b", "neutral"), _pred("c", "neutral")])
    rwsft_repaired = pd.DataFrame([_pred("a", "neutral"), _pred("b", "neutral"), _pred("c", "neutral")])

    contexts_path = tmp_path / "contexts.parquet"
    dpo_original_path = tmp_path / "dpo_original.parquet"
    dpo_repaired_path = tmp_path / "dpo_repaired.parquet"
    rwsft_repaired_path = tmp_path / "rwsft_repaired.parquet"
    output_path = tmp_path / "table.csv"
    metrics_path = tmp_path / "metrics.json"
    status_path = tmp_path / "status.json"
    manifest_path = tmp_path / "manifest.json"
    contexts.to_parquet(contexts_path, index=False)
    dpo_original.to_parquet(dpo_original_path, index=False)
    dpo_repaired.to_parquet(dpo_repaired_path, index=False)
    rwsft_repaired.to_parquet(rwsft_repaired_path, index=False)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "repaired_forecast_baseline_probe_v6",
            "--contexts",
            str(contexts_path),
            "--dpo-original",
            str(dpo_original_path),
            "--dpo-repaired",
            str(dpo_repaired_path),
            "--rwsft-repaired",
            str(rwsft_repaired_path),
            "--output",
            str(output_path),
            "--metrics",
            str(metrics_path),
            "--status",
            str(status_path),
            "--manifest",
            str(manifest_path),
        ],
    )

    assert probe.main() == 0
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert metrics["pipeline_pass"] is True
    assert metrics["claim_allowed"] is False
    assert metrics["dpo_repair_delta_macro_f1"] > 0
    assert metrics["dpo_repaired_beats_rwsft_macro_f1"] is True
    assert output_path.exists()
