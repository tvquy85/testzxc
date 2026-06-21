import json
import sys
from argparse import Namespace

import pandas as pd

from src.alignment.build_dpo_pairs_v2 import build_dpo_pairs
from src.alignment.train_dpo_v2 import load_dpo_records
from src.eval.backtest_daily_portfolio_v2 import main as backtest_main
from src.eval.build_counterfactual_contexts_v2 import build_task
from src.eval.evaluate_counterfactual_directional_v2 import main as counterfactual_main
from src.eval.forecast_prediction import parse_forecast_prediction
from src.eval.generate_test_predictions_v2 import prediction_quality_failures, select_prediction_rows
from src.eval.run_baseline_suite import main as baseline_main


def test_dpo_pairs_include_prompt_messages_and_train_only():
    scored = pd.DataFrame(
        [
            {
                "sample_id": "s1",
                "candidate_id": 0,
                "split": "train",
                "prompt": "Explain this context.",
                "raw_text": "weak answer",
                "alignment_proxy_score": 0.2,
            },
            {
                "sample_id": "s1",
                "candidate_id": 1,
                "split": "train",
                "prompt": "Explain this context.",
                "raw_text": "strong answer",
                "alignment_proxy_score": 0.8,
            },
            {
                "sample_id": "s2",
                "candidate_id": 0,
                "split": "val",
                "prompt": "Do not use.",
                "raw_text": "bad split",
                "alignment_proxy_score": 1.0,
            },
        ]
    )

    pairs = build_dpo_pairs(scored, min_gap=0.1)

    assert len(pairs) == 1
    pair = pairs[0]
    assert pair["split"] == "train"
    assert pair["chosen_score"] > pair["rejected_score"]
    assert pair["prompt"] == "Explain this context."
    assert pair["messages"] == [{"role": "user", "content": "Explain this context."}]


def test_load_dpo_records_requires_prompt_chosen_rejected(tmp_path):
    path = tmp_path / "dpo.jsonl"
    rows = [
        {"sample_id": "ok", "split": "train", "prompt": "p", "chosen": "c", "rejected": "r"},
        {"sample_id": "missing_prompt", "split": "train", "chosen": "c", "rejected": "r"},
        {"sample_id": "bad_split", "split": "test", "prompt": "p", "chosen": "c", "rejected": "r"},
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    records = load_dpo_records(str(path), limit=10)

    assert [record["sample_id"] for record in records] == ["ok"]


def test_backtest_uses_turnover_cost_and_test_only_predictions(tmp_path, monkeypatch):
    labels = pd.DataFrame(
        [
            {"sample_id": "a", "ticker": "AAA", "event_date": "2023-01-02", "split": "test", "stock_return_h1": 0.01},
            {"sample_id": "b", "ticker": "AAA", "event_date": "2023-01-03", "split": "test", "stock_return_h1": -0.02},
        ]
    )
    preds = pd.DataFrame(
        [
            {"sample_id": "a", "split": "test", "action": "long", "pred_label": "strong_up", "schema_ok": True},
            {"sample_id": "b", "split": "test", "action": "short", "pred_label": "strong_down", "schema_ok": True},
        ]
    )
    labels_path = tmp_path / "labels.parquet"
    preds_path = tmp_path / "preds.parquet"
    metrics_path = tmp_path / "metrics.json"
    daily_path = tmp_path / "daily.csv"
    status_path = tmp_path / "status.json"
    manifest_path = tmp_path / "manifest.json"
    labels.to_parquet(labels_path, index=False)
    preds.to_parquet(preds_path, index=False)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "backtest_daily_portfolio_v2.py",
            "--predictions",
            str(preds_path),
            "--labels",
            str(labels_path),
            "--output-json",
            str(metrics_path),
            "--daily-returns",
            str(daily_path),
            "--status",
            str(status_path),
            "--manifest",
            str(manifest_path),
        ],
    )

    assert backtest_main() == 0
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert metrics["total_turnover"] == 3.0
    assert metrics["nonzero_daily_return_rate"] == 1.0
    assert json.loads(status_path.read_text(encoding="utf-8"))["status"] == "PASS"


def test_backtest_medium_gate_rejects_too_few_trading_days(tmp_path, monkeypatch):
    labels = pd.DataFrame(
        [
            {"sample_id": "a", "ticker": "AAA", "event_date": "2023-01-02", "split": "test", "stock_return_h1": 0.01},
            {"sample_id": "b", "ticker": "BBB", "event_date": "2023-01-02", "split": "test", "stock_return_h1": -0.01},
        ]
    )
    preds = pd.DataFrame(
        [
            {"sample_id": "a", "split": "test", "action": "long", "pred_label": "strong_up", "schema_ok": True},
            {"sample_id": "b", "split": "test", "action": "short", "pred_label": "strong_down", "schema_ok": True},
        ]
    )
    labels_path = tmp_path / "labels.parquet"
    preds_path = tmp_path / "preds.parquet"
    labels.to_parquet(labels_path, index=False)
    preds.to_parquet(preds_path, index=False)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "backtest_daily_portfolio_v2.py",
            "--predictions",
            str(preds_path),
            "--labels",
            str(labels_path),
            "--min-trading-days",
            "20",
            "--output-json",
            str(tmp_path / "metrics.json"),
            "--daily-returns",
            str(tmp_path / "daily.csv"),
            "--status",
            str(tmp_path / "status.json"),
            "--manifest",
            str(tmp_path / "manifest.json"),
        ],
    )

    assert backtest_main() == 1
    status = json.loads((tmp_path / "status.json").read_text(encoding="utf-8"))
    assert status["status"] == "FAIL"
    assert any("trading days" in failure for failure in status["failures"])


def test_forecast_only_parser_accepts_valid_and_rejects_malformed():
    valid = json.dumps(
        {
            "forecast_distribution": {
                "Strong Down": 0.1,
                "Mild Down": 0.2,
                "Neutral": 0.3,
                "Mild Up": 0.25,
                "Strong Up": 0.15,
            },
            "action": "long",
        }
    )
    dist, action, pred_label, parse_ok, schema_ok, errors = parse_forecast_prediction(valid)
    assert parse_ok
    assert schema_ok
    assert not errors
    assert abs(sum(dist.values()) - 1.0) < 1e-9
    assert action == "long"
    assert pred_label in {"strong_down", "mild_down", "neutral", "mild_up", "strong_up"}

    _, action, pred_label, parse_ok, schema_ok, errors = parse_forecast_prediction('{"forecast_distribution": {"Neutral": 1.0}}')
    assert parse_ok
    assert not schema_ok
    assert action == "invalid"
    assert pred_label == "invalid"
    assert errors


def test_forecast_only_parser_rejects_inconsistent_action():
    inconsistent = json.dumps(
        {
            "forecast_distribution": {
                "Strong Down": 0.3,
                "Mild Down": 0.2,
                "Neutral": 0.3,
                "Mild Up": 0.1,
                "Strong Up": 0.1,
            },
            "action": "hold",
        }
    )
    _, action, pred_label, parse_ok, schema_ok, errors = parse_forecast_prediction(inconsistent, action_policy="strict")
    assert parse_ok
    assert not schema_ok
    assert action == "invalid"
    assert pred_label == "invalid"
    assert any("action_distribution_inconsistent" in err for err in errors)

    _, action, pred_label, parse_ok, schema_ok, errors = parse_forecast_prediction(inconsistent, action_policy="derive")
    assert parse_ok
    assert schema_ok
    assert action == "short"
    assert pred_label == "strong_down"
    assert not errors


def test_prediction_quality_gate_rejects_all_neutral_hold():
    preds = pd.DataFrame(
        [
            {"sample_id": "a", "split": "test", "action": "hold", "parse_ok": True, "schema_ok": True},
            {"sample_id": "b", "split": "test", "action": "hold", "parse_ok": True, "schema_ok": True},
        ]
    )
    args = Namespace(
        min_parse_ok_rate=0.80,
        min_schema_ok_rate=0.80,
        allow_fallback=False,
        allow_non_dpo_checkpoint=False,
    )

    failures = prediction_quality_failures(preds, args, "checkpoints/aligned/qwen3_4b/dpo_v2", "primary")

    assert any("all schema-valid actions are hold" in failure for failure in failures)


def test_date_aware_prediction_selection_is_chronological_and_test_only():
    samples = pd.DataFrame(
        [
            {"sample_id": "train", "split": "train", "event_date": "2022-12-30", "headline": "", "body": ""},
            {"sample_id": "a1", "split": "test", "event_date": "2023-01-02", "headline": "", "body": ""},
            {"sample_id": "a2", "split": "test", "event_date": "2023-01-02", "headline": "", "body": ""},
            {"sample_id": "b1", "split": "test", "event_date": "2023-01-03", "headline": "", "body": ""},
            {"sample_id": "b2", "split": "test", "event_date": "2023-01-03", "headline": "", "body": ""},
            {"sample_id": "c1", "split": "test", "event_date": "2023-01-04", "headline": "", "body": ""},
        ]
    )
    tokens = pd.DataFrame(
        [
            {"sample_id": "a1", "regime_label": "normal_vol", "technical_event_tokens_json": "[]"},
            {"sample_id": "b1", "regime_label": "normal_vol", "technical_event_tokens_json": "[]"},
        ]
    )
    args = Namespace(
        split="test",
        start_date="2023-01-02",
        end_date=None,
        max_days=2,
        max_rows_per_day=1,
        limit=10,
    )

    selected = select_prediction_rows(samples, tokens, args)

    assert list(selected["sample_id"]) == ["a1", "b1"]
    assert set(selected["split"]) == {"test"}
    assert selected["event_date"].is_monotonic_increasing


def test_counterfactual_task_has_original_counterfactual_and_expected_direction():
    row = {
        "sample_id": "s1",
        "ticker": "AAA",
        "event_date": "2023-01-02",
        "split": "test",
        "headline": "Company faces lawsuit warning",
        "body": "Shares fell after weak guidance.",
        "technical_event_tokens_json": '[{"token":"RSI_OVERBOUGHT","direction_prior":"bearish","strength":"strong"}]',
    }

    task = build_task(row, "remove_bad_news", "less_negative")

    assert task["split"] == "test"
    assert task["expected_direction"] == "less_negative"
    assert "original_headline" in task
    assert "counterfactual_headline" in task
    assert "lawsuit" not in task["counterfactual_headline"].lower()


def test_counterfactual_eval_requires_adapter_and_writes_directional_metrics(tmp_path, monkeypatch):
    checkpoint = tmp_path / "dpo"
    checkpoint.mkdir()
    (checkpoint / "adapter_config.json").write_text("{}", encoding="utf-8")
    (checkpoint / "adapter_model.safetensors").write_bytes(b"adapter")
    tasks = [
        {
            "sample_id": "s1",
            "split": "test",
            "counterfactual_type": "remove_bad_news",
            "expected_direction": "less_negative",
            "original_headline": "lawsuit warning",
            "original_body": "weak results",
            "original_technical_event_tokens_json": "[]",
            "counterfactual_headline": "neutral update",
            "counterfactual_body": "neutral results",
            "counterfactual_technical_event_tokens_json": "[]",
        }
    ]
    input_path = tmp_path / "tasks.jsonl"
    output_path = tmp_path / "metrics.json"
    examples_path = tmp_path / "examples.csv"
    status_path = tmp_path / "status.json"
    manifest_path = tmp_path / "manifest.json"
    input_path.write_text("\n".join(json.dumps(task) for task in tasks) + "\n", encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_counterfactual_directional_v2.py",
            "--checkpoint",
            str(checkpoint),
            "--input",
            str(input_path),
            "--evaluator",
            "heuristic",
            "--debug-allow-heuristic-pass",
            "--output",
            str(output_path),
            "--examples",
            str(examples_path),
            "--status",
            str(status_path),
            "--manifest",
            str(manifest_path),
        ],
    )

    assert counterfactual_main() == 0
    metrics = json.loads(output_path.read_text(encoding="utf-8"))
    assert metrics["num_tasks"] == 1
    assert metrics["pass_rate"] == 1.0
    assert json.loads(status_path.read_text(encoding="utf-8"))["status"] == "PASS"


def test_step18_baseline_suite_cannot_pass_all_not_run(tmp_path, monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_baseline_suite.py",
            "--write-not-run-only",
            "--output",
            str(tmp_path / "baselines.csv"),
            "--aggregate",
            str(tmp_path / "baseline_aggregate.csv"),
            "--summary",
            str(tmp_path / "summary.json"),
            "--predictions-output",
            str(tmp_path / "baseline_predictions.parquet"),
            "--status",
            str(tmp_path / "status.json"),
            "--manifest",
            str(tmp_path / "manifest.json"),
        ],
    )

    assert baseline_main() == 1
    status = json.loads((tmp_path / "status.json").read_text(encoding="utf-8"))
    assert status["status"] == "FAIL"
    assert status["next_step_allowed"] is False
