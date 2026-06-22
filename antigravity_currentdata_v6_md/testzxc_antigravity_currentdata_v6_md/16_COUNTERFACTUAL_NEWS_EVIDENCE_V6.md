# 16 — Evidence-Level Counterfactual Faithfulness V6

> Scope: branch `currentdata-aaai-fix-v2`; current-data only; do not use SN2; do not expand full FNSPID. Execute one file at a time. Every step must write a status JSON and must stop on FAIL. Pipeline PASS is not a scientific claim.

## Goal
Improve and test news faithfulness. Medium counterfactual was blocked because positive/negative news perturbations were weak.

## Outputs
```text
outputs/metrics/16_v6_counterfactual_news_evidence.json
outputs/tables/16_v6_counterfactual_breakdown.csv
review_samples/currentdata_v6/16_counterfactual_failure_samples.jsonl
outputs/status/16_COUNTERFACTUAL_NEWS_EVIDENCE_V6.status.json
```

## Tasks
`remove_positive_evidence`, `remove_negative_evidence`, `neutralize_positive_evidence`, `neutralize_negative_evidence`, `remove_all_company_evidence`, `neutralize_bearish_technical`, `neutralize_bullish_technical`.

## Test case
```python
from src.eval.counterfactual_direction_rules_v6 import expected_direction

def test_expected_direction_positive_removed():
    assert expected_direction('remove_positive_evidence') == 'up_decreases'

def test_expected_direction_negative_removed():
    assert expected_direction('remove_negative_evidence') == 'down_decreases'
```

## Commands
```bash
python -m src.eval.build_counterfactual_evidence_v6 --contexts data/processed/ticker_date_evidence_contexts_h1_v6_repaired.parquet --predictions outputs/predictions/current_v6_dpo_predictions.parquet --output data/eval/current_v6_counterfactual_tasks.jsonl --status outputs/status/16A_BUILD_COUNTERFACTUAL_V6.status.json
python -m src.eval.evaluate_counterfactual_directional_v4 --tasks data/eval/current_v6_counterfactual_tasks.jsonl --adapter outputs/models/qwen3_current_v6_dpo_adapter --config configs/local_paths.yaml --metrics outputs/metrics/16_v6_counterfactual_news_evidence.json --breakdown outputs/tables/16_v6_counterfactual_breakdown.csv --failures review_samples/currentdata_v6/16_counterfactual_failure_samples.jsonl --status outputs/status/16_COUNTERFACTUAL_NEWS_EVIDENCE_V6.status.json
python -m pytest -q tests/test_counterfactual_direction_v6.py tests
```

## Acceptance
Overall pass >=0.50 for claim; no-change <=0.35; remove positive/negative evidence >=0.35; schema >=0.90.

## Progress Update 2026-06-22
Status: `PASS` for pipeline/artifact validity; status file reports `next_step_allowed=true`. Counterfactual/news-faithfulness claim is blocked.

Implementation notes:
```text
src/eval/counterfactual_direction_rules_v6.py
src/eval/build_counterfactual_evidence_v6.py
tests/test_counterfactual_direction_v6.py
```

The V6 builder creates evidence-level tasks from the Step 14 prediction-context artifact, not the smaller original repaired context file, so task samples align with the 300 DPO prediction contexts. Task output is round-robin across task types so smoke/medium stages are representative.

Artifacts verified:
```text
data/eval/current_v6_counterfactual_tasks.jsonl
outputs/metrics/16A_v6_counterfactual_build.json
outputs/status/16A_BUILD_COUNTERFACTUAL_V6.status.json
outputs/metrics/16_v6_counterfactual_news_evidence.json
outputs/tables/16_v6_counterfactual_breakdown.csv
review_samples/currentdata_v6/16_counterfactual_failure_samples.jsonl
outputs/manifests/16_COUNTERFACTUAL_NEWS_EVIDENCE_V6.manifest.json
outputs/status/16_COUNTERFACTUAL_NEWS_EVIDENCE_V6.status.json
```

Build command:
```bash
D:\LOBProj\LOBExp\.venv\Scripts\python.exe -m src.eval.build_counterfactual_evidence_v6 --contexts data/processed/current_v6_prediction_contexts.parquet --predictions outputs/predictions/current_v6_dpo_predictions.parquet --output data/eval/current_v6_counterfactual_tasks.jsonl --metrics outputs/metrics/16A_v6_counterfactual_build.json --status outputs/status/16A_BUILD_COUNTERFACTUAL_V6.status.json --manifest outputs/manifests/16A_BUILD_COUNTERFACTUAL_V6.manifest.json --limit 350 --min-per-type 10
```

Build metrics:
```text
tasks_written: 350
samples_used: 99
context_rows_considered: 300
type_counts: 50 per each of 7 task types
applicable_counts:
  remove_positive_evidence: 273
  remove_negative_evidence: 124
  neutralize_positive_evidence: 273
  neutralize_negative_evidence: 124
  remove_all_company_evidence: 165
  neutralize_bearish_technical: 291
  neutralize_bullish_technical: 245
```

Evaluation command:
```bash
D:\LOBProj\LOBExp\.venv\Scripts\python.exe -m src.eval.evaluate_counterfactual_directional_v4 --tasks data/eval/current_v6_counterfactual_tasks.jsonl --adapter outputs/models/qwen3_current_v6_dpo_adapter --config configs/local_paths.yaml --model-key main_explanation_llm --hf-home E:/huggingface --metrics outputs/metrics/16_v6_counterfactual_news_evidence.json --breakdown outputs/tables/16_v6_counterfactual_breakdown.csv --failures review_samples/currentdata_v6/16_counterfactual_failure_samples.jsonl --status outputs/status/16_COUNTERFACTUAL_NEWS_EVIDENCE_V6.status.json --manifest outputs/manifests/16_COUNTERFACTUAL_NEWS_EVIDENCE_V6.manifest.json --max-tasks 350 --batch-size 8 --max-new-tokens 128 --min-schema-ok-rate 0.90 --min-pass-rate 0.50 --max-no-change-rate 0.35
```

Final metrics:
```text
num_tasks: 350
parse_ok_rate: 1.0000
schema_ok_rate: 0.9771
pair_schema_ok_rate: 0.9543
pass_rate: 0.3257
wrong_direction_rate: 0.3086
no_change_rate: 0.3657
claim_allowed: false
general_counterfactual_claim_allowed: false
news_faithfulness_claim_allowed: false
```

Breakdown:
```text
remove_positive_evidence: pass_rate=0.30, no_change_rate=0.14, schema_ok_rate=0.88
remove_negative_evidence: pass_rate=0.54, no_change_rate=0.12, schema_ok_rate=0.94
neutralize_positive_evidence: pass_rate=0.18, no_change_rate=0.68, schema_ok_rate=1.00
neutralize_negative_evidence: pass_rate=0.14, no_change_rate=0.68, schema_ok_rate=1.00
remove_all_company_evidence: pass_rate=0.28, no_change_rate=0.16, schema_ok_rate=0.86
neutralize_bearish_technical: pass_rate=0.34, no_change_rate=0.38, schema_ok_rate=1.00
neutralize_bullish_technical: pass_rate=0.50, no_change_rate=0.40, schema_ok_rate=1.00
```

Acceptance result:
```text
pipeline/status JSON gate: PASS
schema gate: PASS (0.9771 >= 0.90)
overall faithfulness claim gate: FAIL (0.3257 < 0.50)
no-change claim gate: FAIL (0.3657 > 0.35)
remove_positive_evidence gate: FAIL (0.30 < 0.35)
remove_negative_evidence gate: PASS (0.54 >= 0.35)
claim_allowed: false
```

Verification commands run:
```bash
D:\LOBProj\LOBExp\.venv\Scripts\python.exe -m py_compile src\eval\counterfactual_direction_rules_v6.py src\eval\build_counterfactual_evidence_v6.py src\eval\evaluate_counterfactual_directional_v4.py
D:\LOBProj\LOBExp\.venv\Scripts\python.exe -m pytest -q tests/test_counterfactual_direction_v6.py
D:\LOBProj\LOBExp\.venv\Scripts\python.exe -m src.utils.verify_status --status outputs/status/16A_BUILD_COUNTERFACTUAL_V6.status.json
D:\LOBProj\LOBExp\.venv\Scripts\python.exe -m src.utils.verify_status --status outputs/status/16_COUNTERFACTUAL_NEWS_EVIDENCE_V6.status.json
D:\LOBProj\LOBExp\.venv\Scripts\python.exe -m pytest -q tests/test_counterfactual_direction_v6.py tests/test_backtest_v6.py tests/test_v6_predictions.py
```

Scientific caution: Step 16 improves task construction and validates deterministic counterfactual evaluation, but the DPO model still does not show enough evidence-level faithfulness for a paper claim. The main failure modes are weak sensitivity to neutralization tasks and insufficient response to removing positive evidence.
