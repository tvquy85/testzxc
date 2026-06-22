import src.judges.judge_ensemble_v6 as judge_ensemble_v6
from src.judges.judge_ensemble_v6 import (
    aggregate_results,
    average_distributions,
    evaluate_gate,
    parse_judge_output,
)

def test_average_distributions_sums_to_one():
    out=average_distributions([{'a':0.2,'b':0.8},{'a':0.4,'b':0.6}])
    assert abs(sum(out.values())-1.0)<1e-8
    assert abs(out['a'] - 0.3) < 1e-8


def test_parser_repairs_095_without_raw_schema_pass():
    parsed = parse_judge_output(
        '{"forecast_distribution":{"strong_down":0.65,"mild_down":0.25,"neutral":0.05,"mild_up":0.0,"strong_up":0.0}}'
    )

    assert parsed["parse_ok"] is True
    assert parsed["raw_schema_ok"] is False
    assert parsed["usable_schema_ok"] is True
    assert parsed["schema_ok"] is True
    assert parsed["repaired"] is True
    assert parsed["repair_type"] == "simplex_projection"
    assert parsed["repair_argmax_changed"] is False
    assert abs(sum(parsed["usable_dist"].values()) - 1.0) < 1e-8
    assert parsed["raw_prob_sum"] == 0.9500000000000001


def test_parser_rejects_outside_repair_band():
    parsed = parse_judge_output(
        '{"forecast_distribution":{"strong_down":0.40,"mild_down":0.20,"neutral":0.20,"mild_up":0.05,"strong_up":0.00}}'
    )

    assert parsed["parse_ok"] is True
    assert parsed["raw_schema_ok"] is False
    assert parsed["usable_schema_ok"] is False
    assert parsed["schema_ok"] is False
    assert parsed["error_type"] == "prob_sum_out_of_range"


def test_parser_rejects_repair_that_changes_argmax(monkeypatch):
    monkeypatch.setattr(
        judge_ensemble_v6,
        "project_to_probability_simplex",
        lambda _: [0.0, 0.0, 0.0, 0.0, 1.0],
    )
    parsed = judge_ensemble_v6.parse_judge_output(
        '{"forecast_distribution":{"strong_down":0.93,"mild_down":0.0,"neutral":0.03,"mild_up":0.0,"strong_up":0.0}}'
    )

    assert parsed["raw_schema_ok"] is False
    assert parsed["usable_schema_ok"] is False
    assert parsed["repair_argmax_changed"] is True
    assert parsed["error_type"] == "repair_argmax_changed"


def test_aggregation_uses_usable_distribution():
    repaired = parse_judge_output(
        '{"forecast_distribution":{"strong_down":0.65,"mild_down":0.25,"neutral":0.05,"mild_up":0.0,"strong_up":0.0}}'
    )
    strict = parse_judge_output(
        '{"forecast_distribution":{"strong_down":0.60,"mild_down":0.25,"neutral":0.10,"mild_up":0.05,"strong_up":0.0}}'
    )
    results = {
        "s1_0": {
            "sample_id": "s1",
            "candidate_id": 0,
            "target_label_5": "strong_down",
            "raw_judgments": {"normal": "", "reversed": ""},
            "parse_results": {"normal": repaired, "reversed": strict},
            "prompts": {"normal": "", "reversed": ""},
        }
    }

    rows, _ = aggregate_results(results, ["normal", "reversed"])

    assert len(rows) == 1
    row = rows[0]
    assert row["raw_judge_schema_ok"] is False
    assert row["judge_schema_ok"] is True
    assert row["valid_variant_count"] == 2
    assert row["p_strong_down_normal_usable"] > row["p_strong_down_normal_raw"]
    assert row["p_strong_down"] > 0.60


def test_aggregation_keeps_probabilities_for_partial_valid_variants():
    strict = parse_judge_output(
        '{"forecast_distribution":{"strong_down":0.05,"mild_down":0.20,"neutral":0.50,"mild_up":0.20,"strong_up":0.05}}'
    )
    invalid = parse_judge_output(
        '{"forecast_distribution":{"strong_down":0.05,"mild_down":0.20,"neutral":0.50,"mild_up":0.10,"strong_up":0.05}}'
    )
    results = {
        "s1_0": {
            "sample_id": "s1",
            "candidate_id": 0,
            "target_label_5": "neutral",
            "raw_judgments": {"normal": "", "reversed": "", "stable_random": ""},
            "parse_results": {"normal": strict, "reversed": invalid, "stable_random": strict},
            "prompts": {"normal": "", "reversed": "", "stable_random": ""},
        }
    }

    rows, _ = aggregate_results(results, ["normal", "reversed", "stable_random"])

    row = rows[0]
    assert row["judge_schema_ok"] is False
    assert row["valid_variant_count"] == 2
    assert abs(sum(row[f"p_{k}"] for k in ["strong_down", "mild_down", "neutral", "mild_up", "strong_up"]) - 1.0) < 1e-8
    assert row["true_label_probability_ensemble"] == 0.50


def test_gate_fails_on_low_true_label_probability_even_when_schema_passes():
    failures = evaluate_gate(
        {
            "usable_schema_ok_rate": 1.0,
            "repair_argmax_change_rate": 0.0,
            "repair_rate": 0.2,
            "mean_argmax_consistency_ensemble": 0.95,
            "mean_true_label_probability_ensemble": 0.18,
        }
    )

    assert any("true_label_prob" in failure for failure in failures)


def test_gate_fails_on_low_consistency():
    failures = evaluate_gate(
        {
            "usable_schema_ok_rate": 1.0,
            "repair_argmax_change_rate": 0.0,
            "repair_rate": 0.2,
            "mean_argmax_consistency_ensemble": 0.70,
            "mean_true_label_probability_ensemble": 0.30,
        }
    )

    assert any("consistency" in failure for failure in failures)
