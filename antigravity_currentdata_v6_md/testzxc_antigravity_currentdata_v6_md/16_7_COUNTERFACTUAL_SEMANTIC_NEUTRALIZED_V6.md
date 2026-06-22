# 16.7 - Counterfactual Semantic-Neutralized V6

> Scope: branch `currentdata-aaai-fix-v2`; current-data only; do not use SN2; do not expand full FNSPID. Execute one file at a time. Every step must write a status JSON and must stop on FAIL. Pipeline PASS is not a scientific claim.

## Goal
Repair the Step 16.6 near miss by making neutralization counterfactuals semantically coherent. Step 16.6 showed that quality filtering made removal counterfactuals pass, but `neutralize_negative_evidence` remained weak because token-level replacement produced unnatural contexts such as isolated `neutral update` fragments. Step 16.7 rewrites positive/negative neutralization as explicit neutral company updates without reintroducing polarity terms.

Method basis:
```text
CheckList: capability tests should specify targeted behavior and expected direction.
Contrast Sets: counterfactual perturbations should be meaningful local edits.
Counterfactually Augmented Data: revised examples should remain internally coherent rather than string-corrupted.
```

## Outputs
```text
data/eval/current_v6_counterfactual_semantic_neutralized_tasks.jsonl
outputs/tables/16_7_v6_counterfactual_semantic_candidates.csv
outputs/tables/16_7_v6_counterfactual_semantic_selected.csv
outputs/tables/16_7_v6_counterfactual_semantic_by_type.csv
outputs/metrics/16_7_v6_counterfactual_semantic_neutralized_tasks.json
outputs/status/16_7_COUNTERFACTUAL_SEMANTIC_NEUTRALIZED_TASKS_V6.status.json
outputs/metrics/16_7_v6_counterfactual_semantic_eval.json
outputs/tables/16_7_v6_counterfactual_semantic_breakdown.csv
outputs/tables/16_7_v6_counterfactual_semantic_rows.csv
review_samples/currentdata_v6/16_7_counterfactual_semantic_failures.jsonl
outputs/status/16_7_COUNTERFACTUAL_SEMANTIC_NEUTRALIZED_EVAL_V6.status.json
```

## Commands
```bash
D:\LOBProj\LOBExp\.venv\Scripts\python.exe -m src.eval.build_counterfactual_quality_filtered_v6 --contexts data/processed/current_v6_prediction_contexts.parquet --output data/eval/current_v6_counterfactual_semantic_neutralized_tasks.jsonl --candidate-output outputs/tables/16_7_v6_counterfactual_semantic_candidates.csv --selected-output outputs/tables/16_7_v6_counterfactual_semantic_selected.csv --summary-output outputs/tables/16_7_v6_counterfactual_semantic_by_type.csv --metrics outputs/metrics/16_7_v6_counterfactual_semantic_neutralized_tasks.json --status outputs/status/16_7_COUNTERFACTUAL_SEMANTIC_NEUTRALIZED_TASKS_V6.status.json --manifest outputs/manifests/16_7_COUNTERFACTUAL_SEMANTIC_NEUTRALIZED_TASKS_V6.manifest.json --step-name 16_7_COUNTERFACTUAL_SEMANTIC_NEUTRALIZED_TASKS_V6 --per-type-limit 30 --min-per-type 8 --semantic-neutralization
D:\LOBProj\LOBExp\.venv\Scripts\python.exe -m src.eval.evaluate_counterfactual_directional_v4 --tasks data/eval/current_v6_counterfactual_semantic_neutralized_tasks.jsonl --adapter outputs/models/qwen3_current_v6_dpo_adapter --config configs/local_paths.yaml --model-key main_explanation_llm --hf-home E:/huggingface --output outputs/metrics/16_7_v6_counterfactual_semantic_eval.json --breakdown-output outputs/tables/16_7_v6_counterfactual_semantic_breakdown.csv --failures-output review_samples/currentdata_v6/16_7_counterfactual_semantic_failures.jsonl --rows-output outputs/tables/16_7_v6_counterfactual_semantic_rows.csv --status outputs/status/16_7_COUNTERFACTUAL_SEMANTIC_NEUTRALIZED_EVAL_V6.status.json --manifest outputs/manifests/16_7_COUNTERFACTUAL_SEMANTIC_NEUTRALIZED_EVAL_V6.manifest.json --step-name 16_7_COUNTERFACTUAL_SEMANTIC_NEUTRALIZED_EVAL_V6 --max-tasks 192 --batch-size 8 --max-new-tokens 128 --min-schema-ok-rate 0.90 --min-pass-rate 0.50 --max-no-change-rate 0.35
D:\LOBProj\LOBExp\.venv\Scripts\python.exe -m pytest -q tests/test_counterfactual_quality_filtered_v6.py tests/test_counterfactual_direction_v6.py
```

## Acceptance
```text
task builder status = PASS
LLM eval status = PASS
schema_ok_rate >= 0.90
pass_rate >= 0.50
no_change_rate <= 0.35
remove_positive and remove_negative pass rates each >= 0.35
claim_allowed = true
```

## Progress Update 2026-06-22
Status: `PASS`; counterfactual faithfulness claim is now allowed by the strict V6 gate.

Semantic-neutralized eval:
```text
num_tasks: 192
schema_ok_rate: 0.9948
pair_schema_ok_rate: 0.9896
pass_rate: 0.5313
no_change_rate: 0.2083
wrong_direction_rate: 0.2604
claim_allowed: true
news_faithfulness_claim_allowed: true
```

Breakdown:
```text
remove_positive_evidence: pass_rate=0.5000, no_change_rate=0.1333
remove_negative_evidence: pass_rate=0.7143, no_change_rate=0.0952
neutralize_positive_evidence: pass_rate=0.6000, no_change_rate=0.1667
neutralize_negative_evidence: pass_rate=0.5238, no_change_rate=0.1429
remove_all_company_evidence: pass_rate=0.5000, no_change_rate=0.2333
neutralize_bearish_technical: pass_rate=0.4000, no_change_rate=0.3667
neutralize_bullish_technical: pass_rate=0.5333, no_change_rate=0.2667
```

Interpretation:
```text
The prior failure was partly evaluator/task-construction noise: token-level neutralization corrupted the text. Semantic neutralization produces a coherent counterfactual contrast set and passes the configured V6 faithfulness gates. This opens the counterfactual claim, but strong-accept readiness remains blocked by Flow, forecast superiority, and paper-level alpha.
```
