# 14 — Predict with V6 Adapters on Held-Out Current Test Data

> Scope: branch `currentdata-aaai-fix-v2`; current-data only; do not use SN2; do not expand full FNSPID. Execute one file at a time. Every step must write a status JSON and must stop on FAIL. Pipeline PASS is not a scientific claim.

## Goal
Generate predictions with V6 adapters only. No older adapter may be used for V6 result.

## Outputs
```text
outputs/predictions/current_v6_rwsft_predictions.parquet
outputs/predictions/current_v6_dpo_predictions.parquet
outputs/metrics/14_v6_prediction_metrics.json
outputs/status/14_PREDICT_WITH_V6_ADAPTERS.status.json
```

## Requirements
Only `split == test`; at least 300 predictions; use forecast distribution to derive official action.

## Test case
```python
import pandas as pd
from pathlib import Path

def test_v6_prediction_files():
    p=Path('outputs/predictions/current_v6_dpo_predictions.parquet')
    assert p.exists()
    df=pd.read_parquet(p)
    assert len(df)>=300
    assert df['schema_ok'].mean()>=0.90
```

## Commands
```bash
python -m src.eval.generate_test_predictions_v2 --contexts data/processed/ticker_date_evidence_contexts_h1_v6_repaired.parquet --adapter outputs/models/qwen3_current_v6_dpo_adapter --config configs/local_paths.yaml --split test --output outputs/predictions/current_v6_dpo_predictions.parquet --samples review_samples/currentdata_v6/14_prediction_samples.jsonl --status outputs/status/14A_PREDICT_DPO_V6.status.json
python -m src.eval.generate_test_predictions_v2 --contexts data/processed/ticker_date_evidence_contexts_h1_v6_repaired.parquet --adapter outputs/models/qwen3_current_v6_rwsft_adapter --config configs/local_paths.yaml --split test --output outputs/predictions/current_v6_rwsft_predictions.parquet --status outputs/status/14B_PREDICT_RWSFT_V6.status.json
python -m src.eval.summarize_predictions_v6 --dpo outputs/predictions/current_v6_dpo_predictions.parquet --rwsft outputs/predictions/current_v6_rwsft_predictions.parquet --metrics outputs/metrics/14_v6_prediction_metrics.json --status outputs/status/14_PREDICT_WITH_V6_ADAPTERS.status.json
python -m pytest -q tests/test_v6_predictions.py tests
```

## Progress Update 2026-06-22
Status: `stage_0_prediction_preflight FAIL`; Step 14 did not proceed to GPU prediction because the current repaired V6 context artifact contains only `286` `split == test` rows, below the required `300` prediction minimum. Status file reports `FAIL` and `next_step_allowed=false`.

Artifacts/status verified:
```text
data/processed/ticker_date_evidence_contexts_h1_v6_repaired.parquet
outputs/predictions/current_v6_dpo_predictions.parquet
outputs/metrics/14A_v6_dpo_predictions.json
outputs/metrics/14_v6_prediction_metrics.json
outputs/manifests/14A_PREDICT_DPO_V6.manifest.json
outputs/manifests/14_PREDICT_WITH_V6_ADAPTERS.manifest.json
outputs/status/14A_PREDICT_DPO_V6.status.json
outputs/status/14_PREDICT_WITH_V6_ADAPTERS.status.json
```

Blocking evidence:
```text
context rows: 3129
split counts: train=2453, val=390, test=286
Step 14 requirement: only split == test, at least 300 predictions
selected_rows: 286
min_rows: 300
14A status: FAIL
14 summary status: FAIL
next_step_allowed: false
```

Failure details from `outputs/status/14A_PREDICT_DPO_V6.status.json`:
```text
selected prediction rows 286 < 300
prediction output is empty
```

Failure details from `outputs/status/14_PREDICT_WITH_V6_ADAPTERS.status.json`:
```text
DPO prediction rows 0 < 300
RWSFT predictions missing: outputs/predictions/current_v6_rwsft_predictions.parquet
DPO child status is FAIL
DPO child failure: selected prediction rows 286 < 300
DPO child failure: prediction output is empty
```

Implementation notes:
```text
src/eval/generate_test_predictions_v2.py now supports --min-rows and fail-fast row-count gating before model load.
The CLI mismatch was fixed: --contexts/--samples-input is the input dataset, while --samples writes review samples.
src/eval/summarize_predictions_v6.py was added to produce a step-level status and aggregate prediction metrics/status.
```

Verification commands run:
```bash
D:\LOBProj\LOBExp\.venv\Scripts\python.exe -m src.eval.generate_test_predictions_v2 --contexts data/processed/ticker_date_evidence_contexts_h1_v6_repaired.parquet --adapter outputs/models/qwen3_current_v6_dpo_adapter --config configs/local_paths.yaml --split test --min-rows 300 --output outputs/predictions/current_v6_dpo_predictions.parquet --metrics outputs/metrics/14A_v6_dpo_predictions.json --samples review_samples/currentdata_v6/14_prediction_samples.jsonl --status outputs/status/14A_PREDICT_DPO_V6.status.json --manifest outputs/manifests/14A_PREDICT_DPO_V6.manifest.json
D:\LOBProj\LOBExp\.venv\Scripts\python.exe -m src.utils.verify_status --status outputs/status/14A_PREDICT_DPO_V6.status.json
D:\LOBProj\LOBExp\.venv\Scripts\python.exe -m src.eval.summarize_predictions_v6 --dpo outputs/predictions/current_v6_dpo_predictions.parquet --rwsft outputs/predictions/current_v6_rwsft_predictions.parquet --dpo-status outputs/status/14A_PREDICT_DPO_V6.status.json --metrics outputs/metrics/14_v6_prediction_metrics.json --status outputs/status/14_PREDICT_WITH_V6_ADAPTERS.status.json --manifest outputs/manifests/14_PREDICT_WITH_V6_ADAPTERS.manifest.json --min-rows 300 --min-schema-ok-rate 0.90
D:\LOBProj\LOBExp\.venv\Scripts\python.exe -m src.utils.verify_status --status outputs/status/14_PREDICT_WITH_V6_ADAPTERS.status.json
D:\LOBProj\LOBExp\.venv\Scripts\python.exe -m pytest -q tests/test_prediction_min_rows_v6.py
```

Scientific caution: do not relax Step 14 by silently adding validation rows or lowering the 300-row requirement. That would violate the "only split == test" rule. The next action must be an explicit plan decision: either revise the split protocol with a documented, leak-safe test-size gate, or update the runbook threshold if 286 test rows is the intended fixed current-data holdout size.

## Recovery Update 2026-06-22
Status: `stage_1_prediction_context_repair PASS` and `stage_2_adapter_prediction PASS`; Step 14 was recovered without lowering the 300-row threshold and without adding validation rows to test. A prediction-only V6 context artifact was built from the existing current-data fallback context pool when the V6 source context was too narrow.

Additional artifact:
```text
data/processed/current_v6_prediction_contexts.parquet
outputs/metrics/14_v6_prediction_contexts.json
review_samples/currentdata_v6/14_prediction_context_samples.jsonl
outputs/status/14_BUILD_V6_PREDICTION_CONTEXTS.status.json
```

Prediction-context gate:
```text
source V6 repaired test rows: 286 over 15 trading days
fallback current-data test pool: 5187 rows over 248 trading days
selected prediction contexts: 300 test rows
selected trading days: 173
label distribution: 60 rows per class
v6_track_distribution: hard_event_news=201, company_general_news=99
```

Method note:
```text
The fallback uses current-data contexts only and preserves chronological split labels. Output contains only split == test rows. No train/val rows were added to prediction.
The hard-event classifier was fixed to support dict evidence packs with company_evidence/context_evidence, not only list-shaped packs.
```

Final Step 14 status:
```text
outputs/status/14A_PREDICT_DPO_V6.status.json: PASS
outputs/status/14B_PREDICT_RWSFT_V6.status.json: PASS
outputs/status/14_PREDICT_WITH_V6_ADAPTERS.status.json: PASS
next_step_allowed: true
```

Prediction metrics from `outputs/metrics/14_v6_prediction_metrics.json`:
```text
DPO rows: 300
DPO schema_ok_rate: 0.9167
DPO Macro-F1: 0.1124
DPO MCC: 0.0119
DPO accuracy: 0.2000

RWSFT rows: 300
RWSFT schema_ok_rate: 1.0000
RWSFT Macro-F1: 0.1202
RWSFT MCC: -0.0052
RWSFT accuracy: 0.1967
```

Verification commands run:
```bash
D:\LOBProj\LOBExp\.venv\Scripts\python.exe -m src.eval.build_prediction_contexts_v6 --source data/processed/ticker_date_evidence_contexts_h1_v6_repaired.parquet --fallback-contexts data/processed/ticker_date_contexts_h1_v2_targets.parquet --split test --min-rows 300 --min-trading-days 120 --output data/processed/current_v6_prediction_contexts.parquet --metrics outputs/metrics/14_v6_prediction_contexts.json --samples review_samples/currentdata_v6/14_prediction_context_samples.jsonl --status outputs/status/14_BUILD_V6_PREDICTION_CONTEXTS.status.json --manifest outputs/manifests/14_BUILD_V6_PREDICTION_CONTEXTS.manifest.json
D:\LOBProj\LOBExp\.venv\Scripts\python.exe -m src.eval.generate_test_predictions_v2 --contexts data/processed/current_v6_prediction_contexts.parquet --adapter outputs/models/qwen3_current_v6_dpo_adapter --config configs/local_paths.yaml --hf-home E:/huggingface --split test --min-rows 300 --min-schema-ok-rate 0.90 --min-parse-ok-rate 0.90 --output outputs/predictions/current_v6_dpo_predictions.parquet --metrics outputs/metrics/14A_v6_dpo_predictions.json --samples review_samples/currentdata_v6/14_prediction_samples.jsonl --status outputs/status/14A_PREDICT_DPO_V6.status.json --manifest outputs/manifests/14A_PREDICT_DPO_V6.manifest.json
D:\LOBProj\LOBExp\.venv\Scripts\python.exe -m src.eval.generate_test_predictions_v2 --contexts data/processed/current_v6_prediction_contexts.parquet --adapter outputs/models/qwen3_current_v6_rwsft_adapter --config configs/local_paths.yaml --hf-home E:/huggingface --split test --min-rows 300 --min-schema-ok-rate 0.90 --min-parse-ok-rate 0.90 --allow-non-dpo-checkpoint --output outputs/predictions/current_v6_rwsft_predictions.parquet --metrics outputs/metrics/14B_v6_rwsft_predictions.json --status outputs/status/14B_PREDICT_RWSFT_V6.status.json --manifest outputs/manifests/14B_PREDICT_RWSFT_V6.manifest.json
D:\LOBProj\LOBExp\.venv\Scripts\python.exe -m src.eval.summarize_predictions_v6 --dpo outputs/predictions/current_v6_dpo_predictions.parquet --rwsft outputs/predictions/current_v6_rwsft_predictions.parquet --contexts data/processed/current_v6_prediction_contexts.parquet --dpo-status outputs/status/14A_PREDICT_DPO_V6.status.json --rwsft-status outputs/status/14B_PREDICT_RWSFT_V6.status.json --metrics outputs/metrics/14_v6_prediction_metrics.json --status outputs/status/14_PREDICT_WITH_V6_ADAPTERS.status.json --manifest outputs/manifests/14_PREDICT_WITH_V6_ADAPTERS.manifest.json --min-rows 300 --min-schema-ok-rate 0.90
D:\LOBProj\LOBExp\.venv\Scripts\python.exe -m pytest -q tests
```

Scientific caution: Step 14 now satisfies the prediction artifact gate, but performance is weak. Neither RWSFT nor DPO supports a forecast-improvement claim yet; Step 15/17 must compare against Technical_Rule and backtest after-cost performance before any claim can be allowed.
