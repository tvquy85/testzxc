# 16.6 - Counterfactual Quality-Filtered Tasks V6

> Scope: branch `currentdata-aaai-fix-v2`; current-data only; do not use SN2; do not expand full FNSPID. Execute one file at a time. Every step must write a status JSON and must stop on FAIL. Pipeline PASS is not a scientific claim.

## Goal
Repair the counterfactual evaluation protocol before over-interpreting Step 16 failures. Step 16 showed weak counterfactual faithfulness, but many original tasks were mixed-polarity or used placeholder counterfactual text. Step 16.6 builds a quality-controlled contrast set, then reruns the same deterministic DPO counterfactual evaluator.

Method basis:
```text
CheckList: behavioral tests should target a specific capability and expected direction.
Contrast Sets: perturbations should be small, meaningful, and locally diagnostic.
Counterfactually Augmented Data: counterfactual revisions should remain internally coherent and avoid gratuitous unrelated changes.
```

## Outputs
```text
data/eval/current_v6_counterfactual_quality_filtered_tasks.jsonl
outputs/tables/16_6_v6_counterfactual_quality_candidates.csv
outputs/tables/16_6_v6_counterfactual_quality_selected.csv
outputs/tables/16_6_v6_counterfactual_quality_by_type.csv
outputs/metrics/16_6_v6_counterfactual_quality_filtered_tasks.json
outputs/status/16_6_COUNTERFACTUAL_QUALITY_FILTERED_TASKS_V6.status.json
outputs/metrics/16_6_v6_counterfactual_quality_filtered_eval.json
outputs/tables/16_6_v6_counterfactual_quality_filtered_breakdown.csv
review_samples/currentdata_v6/16_6_counterfactual_quality_filtered_failures.jsonl
outputs/status/16_6_COUNTERFACTUAL_QUALITY_FILTERED_EVAL_V6.status.json
```

## Method
Generate all applicable Step 16 task candidates from the 300 current-data test contexts. Keep news tasks only when the expected polarity is dominant in the original context and removed from the counterfactual context. Keep technical tasks only when the technical token JSON actually changes. Repair placeholder counterfactual text with neutral, explicit context strings that do not reintroduce polarity terms. Select a balanced round-robin JSONL for LLM evaluation.

The DPO adapter is evaluated with the same `evaluate_counterfactual_directional_v4` logic, deterministic temperature `0.0`, and `min_delta=0.03`.

## Commands
```bash
D:\LOBProj\LOBExp\.venv\Scripts\python.exe -m src.eval.build_counterfactual_quality_filtered_v6 --contexts data/processed/current_v6_prediction_contexts.parquet --output data/eval/current_v6_counterfactual_quality_filtered_tasks.jsonl --candidate-output outputs/tables/16_6_v6_counterfactual_quality_candidates.csv --selected-output outputs/tables/16_6_v6_counterfactual_quality_selected.csv --summary-output outputs/tables/16_6_v6_counterfactual_quality_by_type.csv --metrics outputs/metrics/16_6_v6_counterfactual_quality_filtered_tasks.json --status outputs/status/16_6_COUNTERFACTUAL_QUALITY_FILTERED_TASKS_V6.status.json --manifest outputs/manifests/16_6_COUNTERFACTUAL_QUALITY_FILTERED_TASKS_V6.manifest.json --per-type-limit 30 --min-per-type 8
D:\LOBProj\LOBExp\.venv\Scripts\python.exe -m src.eval.evaluate_counterfactual_directional_v4 --tasks data/eval/current_v6_counterfactual_quality_filtered_tasks.jsonl --adapter outputs/models/qwen3_current_v6_dpo_adapter --config configs/local_paths.yaml --model-key main_explanation_llm --hf-home E:/huggingface --output outputs/metrics/16_6_v6_counterfactual_quality_filtered_eval.json --breakdown-output outputs/tables/16_6_v6_counterfactual_quality_filtered_breakdown.csv --failures-output review_samples/currentdata_v6/16_6_counterfactual_quality_filtered_failures.jsonl --status outputs/status/16_6_COUNTERFACTUAL_QUALITY_FILTERED_EVAL_V6.status.json --manifest outputs/manifests/16_6_COUNTERFACTUAL_QUALITY_FILTERED_EVAL_V6.manifest.json --step-name 16_6_COUNTERFACTUAL_QUALITY_FILTERED_EVAL_V6 --max-tasks 192 --batch-size 8 --max-new-tokens 128 --min-schema-ok-rate 0.90 --min-pass-rate 0.50 --max-no-change-rate 0.35
D:\LOBProj\LOBExp\.venv\Scripts\python.exe -m pytest -q tests/test_counterfactual_quality_filtered_v6.py tests/test_counterfactual_direction_v6.py
```

## Acceptance
```text
quality-filtered task builder status = PASS
LLM eval status = PASS
schema_ok_rate >= 0.90
claim_allowed only if pass_rate >= 0.50 and no_change_rate <= 0.35
news_faithfulness_claim_allowed only if remove_positive and remove_negative pass rates are each >= 0.35
```

## Progress Update 2026-06-22
Status: `PASS` for task builder and LLM evaluation. Counterfactual claim remains blocked, but the failure is now much narrower.

Builder metrics:
```text
candidate_tasks: 1495
quality_pass_tasks: 1217
selected_tasks: 192
selected_sample_count: 83
original_placeholder_candidate_rate: 0.3759
placeholder_repaired_selected_rate: 0.4219
```

Full quality-filtered LLM eval:
```text
num_tasks: 192
schema_ok_rate: 0.9974
pair_schema_ok_rate: 0.9948
pass_rate: 0.4844
no_change_rate: 0.2604
wrong_direction_rate: 0.2552
claim_allowed: false
news_faithfulness_claim_allowed: true
```

Breakdown:
```text
remove_positive_evidence: pass_rate=0.5000, no_change_rate=0.1333
remove_negative_evidence: pass_rate=0.7143, no_change_rate=0.0952
neutralize_positive_evidence: pass_rate=0.6000, no_change_rate=0.3000
neutralize_negative_evidence: pass_rate=0.0952, no_change_rate=0.4286
remove_all_company_evidence: pass_rate=0.5000, no_change_rate=0.2333
neutralize_bearish_technical: pass_rate=0.4000, no_change_rate=0.3667
neutralize_bullish_technical: pass_rate=0.5333, no_change_rate=0.2667
```

Interpretation:
```text
Quality filtering repairs the task protocol enough to pass the news removal gates and reduce no-change well below the original Step 16 result. The overall pass rate is still just below the required 0.50, mainly because neutralize_negative_evidence remains weak. This points to contrastive evidence-sensitivity training or a pairwise counterfactual objective, not a claim promotion.
```

Verification commands run:
```bash
D:\LOBProj\LOBExp\.venv\Scripts\python.exe -m py_compile src\eval\build_counterfactual_quality_filtered_v6.py src\eval\evaluate_counterfactual_directional_v4.py
D:\LOBProj\LOBExp\.venv\Scripts\python.exe -m pytest -q tests\test_counterfactual_quality_filtered_v6.py tests\test_counterfactual_direction_v6.py
D:\LOBProj\LOBExp\.venv\Scripts\python.exe -m src.utils.verify_status --status outputs/status/16_6_COUNTERFACTUAL_QUALITY_FILTERED_TASKS_V6.status.json
D:\LOBProj\LOBExp\.venv\Scripts\python.exe -m src.utils.verify_status --status outputs/status/16_6_COUNTERFACTUAL_QUALITY_FILTERED_EVAL_V6.status.json
```
