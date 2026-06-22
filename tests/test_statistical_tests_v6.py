import json
import sys

import pandas as pd

from src.repro import run_v6_statistical_tests as stats


def test_moving_block_indices_wraps_and_preserves_length():
    rng = stats.np.random.default_rng(7)
    idx = stats.moving_block_indices(5, 3, rng)
    assert len(idx) == 5
    assert set(idx.tolist()) <= set(range(5))


def test_mcnemar_one_sided_penalizes_candidate_with_fewer_unique_wins():
    candidate = stats.np.array([True, False, False, False])
    benchmark = stats.np.array([False, True, True, False])
    p_value, cand_only, bench_only = stats.mcnemar_one_sided_pvalue(candidate, benchmark)
    assert cand_only == 1
    assert bench_only == 2
    assert p_value > 0.5


def test_statistical_tests_end_to_end(tmp_path, monkeypatch):
    contexts = pd.DataFrame(
        [
            {
                "sample_id": "s1",
                "split": "test",
                "ticker": "AAA",
                "event_date": "2024-01-02",
                "target_return": 0.02,
                "target_label_5": "mild_up",
                "technical_event_tokens_json": json.dumps(
                    [{"direction_prior": "bullish_momentum", "strength": "strong"}]
                ),
            },
            {
                "sample_id": "s2",
                "split": "test",
                "ticker": "BBB",
                "event_date": "2024-01-03",
                "target_return": -0.02,
                "target_label_5": "mild_down",
                "technical_event_tokens_json": json.dumps(
                    [{"direction_prior": "bearish_momentum", "strength": "strong"}]
                ),
            },
            {
                "sample_id": "s3",
                "split": "test",
                "ticker": "CCC",
                "event_date": "2024-01-04",
                "target_return": 0.00,
                "target_label_5": "neutral",
                "technical_event_tokens_json": "[]",
            },
            {
                "sample_id": "s4",
                "split": "test",
                "ticker": "DDD",
                "event_date": "2024-01-05",
                "target_return": 0.03,
                "target_label_5": "strong_up",
                "technical_event_tokens_json": json.dumps(
                    [{"direction_prior": "bearish_momentum", "strength": "strong"}]
                ),
            },
        ]
    )
    dpo = pd.DataFrame(
        [
            {"sample_id": "s1", "schema_ok": True, "pred_label": "mild_up"},
            {"sample_id": "s2", "schema_ok": True, "pred_label": "mild_down"},
            {"sample_id": "s3", "schema_ok": True, "pred_label": "neutral"},
            {"sample_id": "s4", "schema_ok": True, "pred_label": "strong_up"},
        ]
    )
    rwsft = pd.DataFrame(
        [
            {"sample_id": "s1", "schema_ok": True, "pred_label": "neutral"},
            {"sample_id": "s2", "schema_ok": True, "pred_label": "neutral"},
            {"sample_id": "s3", "schema_ok": True, "pred_label": "neutral"},
            {"sample_id": "s4", "schema_ok": True, "pred_label": "neutral"},
        ]
    )
    daily = pd.DataFrame(
        [
            {"date": "2024-01-02", "daily_return_net": 0.01},
            {"date": "2024-01-03", "daily_return_net": 0.02},
            {"date": "2024-01-04", "daily_return_net": 0.00},
            {"date": "2024-01-05", "daily_return_net": 0.03},
        ]
    )
    backtest_metrics = {
        "cost_bps": 0.0,
        "slippage_bps": 0.0,
        "short_borrow_bps": 0.0,
    }

    contexts_path = tmp_path / "contexts.parquet"
    dpo_path = tmp_path / "dpo.parquet"
    rwsft_path = tmp_path / "rwsft.parquet"
    daily_path = tmp_path / "daily.csv"
    backtest_path = tmp_path / "backtest.json"
    output_path = tmp_path / "tests.csv"
    daily_comparison_path = tmp_path / "daily_compare.csv"
    metrics_path = tmp_path / "metrics.json"
    status_path = tmp_path / "status.json"
    manifest_path = tmp_path / "manifest.json"

    contexts.to_parquet(contexts_path)
    dpo.to_parquet(dpo_path)
    rwsft.to_parquet(rwsft_path)
    daily.to_csv(daily_path, index=False)
    backtest_path.write_text(json.dumps(backtest_metrics), encoding="utf-8")

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_v6_statistical_tests",
            "--daily-returns",
            str(daily_path),
            "--contexts",
            str(contexts_path),
            "--dpo-predictions",
            str(dpo_path),
            "--rwsft-predictions",
            str(rwsft_path),
            "--backtest-metrics",
            str(backtest_path),
            "--output",
            str(output_path),
            "--daily-comparison-output",
            str(daily_comparison_path),
            "--metrics",
            str(metrics_path),
            "--status",
            str(status_path),
            "--manifest",
            str(manifest_path),
            "--n-bootstrap",
            "25",
            "--n-permutations",
            "25",
            "--block-size",
            "2",
        ],
    )

    assert stats.main() == 0
    table = pd.read_csv(output_path)
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert "dpo_sharpe_ci" in set(table["test_id"])
    assert "dpo_vs_technical_macro_f1_delta_ci" in set(table["test_id"])
    assert metrics["required_tests_present"] is True
    assert metrics["pipeline_pass"] is True
