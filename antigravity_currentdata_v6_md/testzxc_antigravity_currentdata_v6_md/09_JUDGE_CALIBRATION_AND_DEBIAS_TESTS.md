# 09 — Judge Calibration and Debias Tests

> Scope: branch `currentdata-aaai-fix-v2`; current-data only; do not use SN2; do not expand full FNSPID. Execute one file at a time. Every step must write a status JSON and must stop on FAIL. Pipeline PASS is not a scientific claim.

## Goal
Quantify whether judge probabilities are meaningful and label-order robust.

## Outputs
```text
outputs/metrics/09_v6_judge_calibration.json
outputs/tables/09_v6_judge_reliability_bins.csv
outputs/status/09_JUDGE_CALIBRATION_AND_DEBIAS_TESTS.status.json
```

## Metrics
Brier score, ECE, NLL if valid, true-label prob by track, argmax consistency by track, label-order KL by track.

## Test case
```python
from src.judges.judge_calibration_v6 import expected_calibration_error

def test_ece_perfect():
    ece=expected_calibration_error([0.9,0.8],[1,1],n_bins=2)
    assert 0 <= ece <= 0.2
```

## Commands
```bash
python -m src.judges.judge_calibration_v6 --input data/judges/current_v6_independent_judge_ensemble.parquet --metrics outputs/metrics/09_v6_judge_calibration.json --bins outputs/tables/09_v6_judge_reliability_bins.csv --status outputs/status/09_JUDGE_CALIBRATION_AND_DEBIAS_TESTS.status.json
python -m pytest -q tests/test_judge_calibration_v6.py tests
```

## Acceptance
All probabilities finite and sum to one. If true-label probability <=0.20, stop Flow/DPO and fix judge or data.

## Method Basis
Use calibration and probabilistic-forecast metrics rather than only argmax accuracy:

- ECE/reliability bins follow the calibration framing in Guo et al., 2017: https://arxiv.org/abs/1706.04599.
- Brier/NLL are proper scoring-rule style checks for probabilistic forecasts; see the proper scoring rules overview and Brier discussion: https://arxiv.org/html/2504.01781v1.
- Label-order robustness is required because LLM judges can be sensitive to option/order permutations; see ICLR 2024 selection-bias work and NAACL Findings 2024 option-order sensitivity: https://openreview.net/forum?id=shr9PXz7T0 and https://aclanthology.org/2024.findings-naacl.130/.

## Progress Update 2026-06-22
Status: `stage_3_full_scale PASS`; input scale is the Step 08 `3000 candidate rows x 3 label-order variants` judge-call pass. Status file reports `PASS` and `next_step_allowed=true`.

Artifacts verified:
```text
outputs/metrics/09_v6_judge_calibration.json
outputs/tables/09_v6_judge_reliability_bins.csv
outputs/manifests/09_JUDGE_CALIBRATION_AND_DEBIAS_TESTS.manifest.json
outputs/status/09_JUDGE_CALIBRATION_AND_DEBIAS_TESTS.status.json
```

Gate metrics from `outputs/metrics/09_v6_judge_calibration.json`:
```text
rows: 3000
calibration_rows: 3000
fully_permuted_rows: 2980
fully_permuted_row_rate: 0.9933
valid_probability_row_rate: 1.000
probability_sum_min: 0.9999999999999999
probability_sum_max: 1.0000000000000002
mean_true_label_probability: 0.2412
top_label_accuracy: 0.257
brier_score_multiclass: 0.8567
nll: 1.6738
ece_top_label: 0.2396
ece_macro_ovr: 0.1284
mean_argmax_consistency_ensemble: 0.9569
mean_label_order_kl: 0.00904
```

Acceptance result:
```text
probability finite/sum gate: PASS (3000/3000 rows valid)
true-label probability gate: PASS (0.2412 > 0.20)
label-order coverage gate: PASS (0.9933 >= 0.95)
consistency carry-forward gate: PASS (0.9569 >= 0.75)
status JSON gate: PASS
test suite: PASS with D:\LOBProj\LOBExp\.venv\Scripts\python.exe
```

Reliability-bin observation: most judge confidences are concentrated in the 0.4-0.6 range, with top-label ECE `0.2396`. This means calibration is usable for a full current-data pipeline gate, but not strong enough to claim a well-calibrated judge.

Track-level caution:
```text
strong_down: true-label probability 0.2792, top-label accuracy 0.438
mild_down: true-label probability 0.2187, top-label accuracy 0.088
neutral: true-label probability 0.2899, top-label accuracy 0.387
mild_up: true-label probability 0.1706, top-label accuracy 0.129
strong_up: true-label probability 0.0566, top-label accuracy 0.000
```

Interpretation: Step 09 permits continuing to Step 10 because the aggregate judge signal is above the `0.20` stop threshold and label-order consistency is high. It does not support a strong standalone judge-quality claim. Downstream Flow/DPO should treat tail labels as weak and preserve `claim_allowed=false` unless later held-out utility and prediction gates improve.

Verification commands run:
```bash
D:\LOBProj\LOBExp\.venv\Scripts\python.exe -m pytest -q tests/test_judge_calibration_v6.py
D:\LOBProj\LOBExp\.venv\Scripts\python.exe -m pytest -q tests/test_judge_ensemble_v6.py tests/test_judge_calibration_v6.py
D:\LOBProj\LOBExp\.venv\Scripts\python.exe -m src.judges.judge_calibration_v6 --input data/judges/current_v6_independent_judge_ensemble.parquet --metrics outputs/metrics/09_v6_judge_calibration.json --bins outputs/tables/09_v6_judge_reliability_bins.csv --status outputs/status/09_JUDGE_CALIBRATION_AND_DEBIAS_TESTS.status.json
D:\LOBProj\LOBExp\.venv\Scripts\python.exe -m src.utils.verify_status --status outputs/status/09_JUDGE_CALIBRATION_AND_DEBIAS_TESTS.status.json
D:\LOBProj\LOBExp\.venv\Scripts\python.exe -m pytest -q tests
```
