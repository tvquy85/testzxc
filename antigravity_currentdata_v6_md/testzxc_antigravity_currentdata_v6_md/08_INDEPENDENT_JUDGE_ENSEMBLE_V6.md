# 08 — Independent Judge Ensemble V6

> Scope: branch `currentdata-aaai-fix-v2`; current-data only; do not use SN2; do not expand full FNSPID. Execute one file at a time. Every step must write a status JSON and must stop on FAIL. Pipeline PASS is not a scientific claim.

## Goal
Strengthen judge signal. Use at least normal, reversed, and stable-random label orders; optionally add a second local judge.

## Outputs
```text
data/judges/current_v6_independent_judge_ensemble.parquet
outputs/metrics/08_v6_judge_ensemble.json
outputs/status/08_INDEPENDENT_JUDGE_ENSEMBLE_V6.status.json
```

## Required columns
`p_strong_down`, `p_mild_down`, `p_neutral`, `p_mild_up`, `p_strong_up`, `true_label_probability_ensemble`, `argmax_consistency_ensemble`, `label_order_kl_mean`, `judge_disagreement_entropy`, `judge_schema_ok`.

## Test case
```python
from src.judges.judge_ensemble_v6 import average_distributions

def test_average_distributions_sums_to_one():
    out=average_distributions([{'a':0.2,'b':0.8},{'a':0.4,'b':0.6}])
    assert abs(sum(out.values())-1.0)<1e-8
    assert out['a']==0.3
```

## Commands
```bash
python -m src.judges.judge_ensemble_v6 --rationales data/rationales/parsed/current_v6_train_qwen3_1000x3_repaired.parquet --contexts data/processed/ticker_date_evidence_contexts_h1_v6_repaired.parquet --config configs/local_paths.yaml --model-key judge_llm --label-order-variants normal,reversed,stable_random --batch-size 4 --max-new-tokens 160 --temperature 0.0 --output data/judges/current_v6_independent_judge_ensemble.parquet --metrics outputs/metrics/08_v6_judge_ensemble.json --samples review_samples/currentdata_v6/08_judge_ensemble_samples.jsonl --status outputs/status/08_INDEPENDENT_JUDGE_ENSEMBLE_V6.status.json
python -m pytest -q tests/test_judge_ensemble_v6.py tests
```

## Acceptance
Schema >=0.95, true-label probability >0.22 minimum, consistency >=0.75.

## Progress Update 2026-06-22
Status: `stage_3_full_scale PASS`; executed scale is `3000 candidate rows x 3 label-order variants = 9000 raw judge calls`. Status file reports `PASS` and `next_step_allowed=true`.

Scale clarification: verified final persisted artifact is not `100x3`; it is `1000 samples x 3 candidates/sample = 3000 candidate rows`, with each candidate judged under 3 label-order variants. Therefore the judge-call scale is `3000 x 3 = 9000` raw judge calls. Any earlier `300x3` checkpoint was an intermediate scale, not the final Step 08 artifact.

Artifacts verified:
```text
data/judges/current_v6_independent_judge_ensemble.parquet
outputs/metrics/08_v6_judge_ensemble.json
review_samples/currentdata_v6/08_judge_ensemble_samples.jsonl
outputs/manifests/08_INDEPENDENT_JUDGE_ENSEMBLE_V6.manifest.json
outputs/status/08_INDEPENDENT_JUDGE_ENSEMBLE_V6.status.json
```

Scale and coverage:
```text
input rationales: 3000 rows, 1000 samples, 3 candidates/sample
input contexts: 3129 samples
judge output: 3000 candidate rows
base ticker-date samples covered: 1000 samples, 3 candidate rows/sample
judge-call scale: 3000 candidate rows x 3 label-order variants = 9000 raw judge calls
```

Recheck note for `300x3` vs `100x3`: verified again from parquet, metrics, status JSON, and raw tmp JSONL. Step 08 V6 is definitely not `100x3`. The persisted final artifact is larger than the `300x3` shorthand checkpoint: `1000 samples x 3 candidates/sample = 3000 candidate rows`, then `3000 candidate rows x 3 label-order variants = 9000 raw judge calls`. The raw tmp JSONL has exactly 9000 lines.

Gate metrics from `outputs/metrics/08_v6_judge_ensemble.json`:
```text
raw_schema_ok_rate: 0.9933
usable_schema_ok_rate: 0.9933
raw_variant_schema_ok_rate: 0.9977
usable_variant_schema_ok_rate: 0.9977
mean_argmax_consistency_ensemble: 0.9569
mean_true_label_probability_ensemble: 0.2412
repair_rate: 0.000
repair_argmax_change_rate: 0.000
error_type_counts: none=8979, prob_sum_out_of_range=21
aggregate_probability_sum_ok_rows: 3000/3000
valid_variant_count_distribution: {1: 1, 2: 19, 3: 2980}
```

Acceptance result:
```text
schema gate: PASS (0.9933 >= 0.95)
true-label probability gate: PASS (0.2412 > 0.22)
consistency gate: PASS (0.9569 >= 0.75)
status JSON gate: PASS
unit tests: PASS with D:\LOBProj\LOBExp\.venv\Scripts\python.exe
```

Verification commands run:
```bash
D:\LOBProj\LOBExp\.venv\Scripts\python.exe -m src.judges.judge_ensemble_v6 --rationales data/rationales/parsed/current_v6_train_qwen3_1000x3_repaired.parquet --contexts data/processed/ticker_date_evidence_contexts_h1_v6_repaired.parquet --config configs/local_paths.yaml --model-key judge_llm --label-order-variants normal,reversed,stable_random --batch-size 4 --max-new-tokens 160 --temperature 0.0 --output data/judges/current_v6_independent_judge_ensemble.parquet --metrics outputs/metrics/08_v6_judge_ensemble.json --samples review_samples/currentdata_v6/08_judge_ensemble_samples.jsonl --status outputs/status/08_INDEPENDENT_JUDGE_ENSEMBLE_V6.status.json --resume
D:\LOBProj\LOBExp\.venv\Scripts\python.exe -m pytest -q tests/test_judge_ensemble_v6.py
D:\LOBProj\LOBExp\.venv\Scripts\python.exe -m src.utils.verify_status --status outputs/status/08_INDEPENDENT_JUDGE_ENSEMBLE_V6.status.json
```

Post-Step-09 precheck repair: aggregation now averages every usable label-order variant instead of writing an all-zero aggregate when one permutation is invalid. `judge_schema_ok` still requires all three variants, so permutation failures remain visible while aggregate probabilities stay valid for calibration. Regression test: `test_aggregation_keeps_probabilities_for_partial_valid_variants`.

Environment note: default `python` resolves to `C:\Python\Python311\python.exe`, where the targeted test is blocked by a `transformers` / `huggingface-hub` version mismatch. No package install was performed; verification used the existing `D:\LOBProj\LOBExp\.venv` environment per AGENTS.md.

Scientific caution: full Step 08 coverage is now complete, but this still proves only judge-pipeline validity. It does not by itself support Flow/DPO or forecast claims; those remain gated by Steps 10+.
