import numpy as np
import pandas as pd

from src.judges.judge_calibration_v6 import (
    PROB_COLS,
    brier_score_multiclass,
    compute_calibration,
    expected_calibration_error,
    reliability_bins,
    validate_probabilities,
)


def test_ece_perfect():
    ece = expected_calibration_error([0.9, 0.8], [1, 1], n_bins=2)
    assert 0 <= ece <= 0.2


def test_brier_score_multiclass_perfect_is_zero():
    probs = np.eye(3)
    y_true = np.array([0, 1, 2])

    assert brier_score_multiclass(probs, y_true) == 0.0


def test_reliability_bins_weighted_gap_matches_ece():
    confidences = np.array([0.9, 0.8, 0.2, 0.1])
    outcomes = np.array([1, 1, 0, 0])

    bins = reliability_bins(confidences, outcomes, n_bins=2)

    assert abs(bins["weighted_abs_gap"].sum() - expected_calibration_error(confidences, outcomes, n_bins=2)) < 1e-12


def test_validate_probabilities_detects_bad_sum():
    df = pd.DataFrame([{col: 0.2 for col in PROB_COLS}])
    df.loc[0, "p_neutral"] = 0.1

    out = validate_probabilities(df)

    assert out["all_probabilities_finite"] is True
    assert out["all_probability_sums_ok"] is False
    assert out["valid_probability_row_rate"] == 0.0


def test_compute_calibration_passes_minimal_valid_frame():
    df = pd.DataFrame(
        [
            {
                "target_label_5": "neutral",
                "p_strong_down": 0.05,
                "p_mild_down": 0.10,
                "p_neutral": 0.70,
                "p_mild_up": 0.10,
                "p_strong_up": 0.05,
                "argmax_consistency_ensemble": 1.0,
                "label_order_kl_mean": 0.0,
                "judge_disagreement_entropy": 0.0,
                "judge_schema_ok": True,
            },
            {
                "target_label_5": "mild_up",
                "p_strong_down": 0.05,
                "p_mild_down": 0.10,
                "p_neutral": 0.10,
                "p_mild_up": 0.70,
                "p_strong_up": 0.05,
                "argmax_consistency_ensemble": 1.0,
                "label_order_kl_mean": 0.0,
                "judge_disagreement_entropy": 0.0,
                "judge_schema_ok": True,
            },
        ]
    )

    metrics, bins, failures = compute_calibration(df, n_bins=2)

    assert failures == []
    assert metrics["rows"] == 2
    assert metrics["calibration_rows"] == 2
    assert metrics["mean_true_label_probability"] == 0.70
    assert len(bins) == 2
