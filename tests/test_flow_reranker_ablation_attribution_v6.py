import json
import sys

import pandas as pd

from src.reward import flow_reranker_ablation_attribution_v6 as attribution


def _row(sample_id: str, candidate_id: int, split: str, utility: float, proxy: float) -> dict:
    return {
        "sample_id": sample_id,
        "candidate_id": candidate_id,
        "split": split,
        "flow_reward_score": 0.5,
        "proxy_average_reward": proxy,
        "single_best_judge_reward": proxy,
        "news_grounding_score": 1.0,
        "technical_grounding_score": 1.0,
        "unsupported_news_claim_rate": 0.0,
        "raw_realized_utility": utility,
    }


def test_flow_reranker_ablation_blocks_flow_attribution_when_no_flow_matches(tmp_path, monkeypatch):
    rows = []
    for idx in range(8):
        split = "train" if idx < 5 else "val"
        rows.extend(
            [
                _row(f"s{idx}", 0, split, 0.02, 0.9),
                _row(f"s{idx}", 1, split, -0.01, 0.1),
            ]
        )
    predictions = pd.DataFrame(rows)
    predictions_path = tmp_path / "flow_predictions.csv"
    output_path = tmp_path / "ablation.csv"
    grid_path = tmp_path / "grid.csv"
    metrics_path = tmp_path / "metrics.json"
    status_path = tmp_path / "status.json"
    manifest_path = tmp_path / "manifest.json"
    predictions.to_csv(predictions_path, index=False)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "flow_reranker_ablation_attribution_v6",
            "--predictions",
            str(predictions_path),
            "--output",
            str(output_path),
            "--grid-output",
            str(grid_path),
            "--metrics",
            str(metrics_path),
            "--status",
            str(status_path),
            "--manifest",
            str(manifest_path),
            "--ridge-alpha-grid",
            "0.1,1",
            "--pairwise-c-grid",
            "0.1,1",
            "--min-train-rows",
            "2",
            "--min-eval-rows",
            "2",
        ],
    )

    assert attribution.main() == 0
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert metrics["pipeline_pass"] is True
    assert metrics["claim_allowed"] is False
    assert metrics["flow_attribution_supported"] is False
    assert metrics["no_flow_matches_or_exceeds_full"] is True
    selected = pd.read_csv(output_path)
    assert {"full", "no_flow", "only_flow"}.issubset(set(selected["feature_set"]))
    assert grid_path.exists()
