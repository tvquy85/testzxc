import json
import sys

import pandas as pd

from src.reward import diagnose_flow_utility_v6 as diag


def test_pair_anatomy_counts_only_utility_varying_pairs():
    df = pd.DataFrame(
        [
            {"sample_id": "a", "raw_realized_utility": 0.1},
            {"sample_id": "a", "raw_realized_utility": 0.1},
            {"sample_id": "a", "raw_realized_utility": 0.2},
            {"sample_id": "b", "raw_realized_utility": 0.0},
            {"sample_id": "b", "raw_realized_utility": 0.0},
        ]
    )
    out = diag.pair_anatomy(df)
    assert out["candidate_pair_count"] == 4
    assert out["utility_varying_pair_count"] == 2
    assert out["samples_with_utility_variation"] == 1


def test_flow_utility_diagnostic_end_to_end(tmp_path, monkeypatch):
    rows = []
    for sample_id in range(120):
        for cand in range(3):
            utility = float(cand - 1) / 100.0
            rows.append(
                {
                    "sample_id": f"s{sample_id}",
                    "candidate_id": cand,
                    "split": "val",
                    "raw_realized_utility": utility,
                    "technical_rule_delta": utility,
                    "news_grounding_score": 1.0,
                    "technical_grounding_score": 1.0,
                    "unsupported_news_claim_rate": 0.0,
                    "flow_reward_score": float(cand),
                    "proxy_average_reward": float(2 - cand),
                    "single_best_judge_reward": float(cand == 2),
                    "technical_rule_reward": utility,
                }
            )
    input_path = tmp_path / "flow.csv"
    metrics_path = tmp_path / "metrics.json"
    summary_path = tmp_path / "summary.csv"
    overlap_path = tmp_path / "overlap.csv"
    quantile_path = tmp_path / "quantile.csv"
    examples_path = tmp_path / "examples.jsonl"
    status_path = tmp_path / "status.json"
    manifest_path = tmp_path / "manifest.json"
    pd.DataFrame(rows).to_csv(input_path, index=False)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "diagnose_flow_utility_v6",
            "--predictions",
            str(input_path),
            "--metrics",
            str(metrics_path),
            "--summary-table",
            str(summary_path),
            "--overlap-table",
            str(overlap_path),
            "--quantile-table",
            str(quantile_path),
            "--examples",
            str(examples_path),
            "--status",
            str(status_path),
            "--manifest",
            str(manifest_path),
        ],
    )

    assert diag.main() == 0
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert metrics["eval_rows"] == 360
    assert metrics["utility_varying_pair_rate"] == 1.0
    assert metrics["flow_rank_win_vs_proxy"] is True
    assert summary_path.exists()
    assert overlap_path.exists()
