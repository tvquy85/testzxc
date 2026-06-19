# 10 — Multi-model proxy judges with bias controls

## Goal

Upgrade inferability judging from one weak parser to a multi-judge, debiased evaluation pipeline.

## Existing file to inspect

```text
src/judges/inferability_judge.py
```

## Codex task

Create:

```text
src/judges/inferability_judge_v2.py
src/judges/run_multi_judge_inferability.py
```

## Required judges

Run at least two local judges if available:

```text
Llama-3-8B-Instruct
Qwen3-4B-Instruct or Qwen2.5-3B-Instruct
DeepSeek-R1-Distill-Qwen-1.5B
```

Use available model paths from config. If one model is missing, mark it `missing` in status, do not silently substitute.

## Bias controls

1. Use deterministic generation:

```text
temperature=0.0
do_sample=False
```

2. Use JSON-only constrained prompt.
3. Parse failure must produce `parse_ok=false`, not default neutral.
4. Add prompt variants:
   - normal label order;
   - reversed label order.
5. Report judge stability:

```text
label_order_consistency
parse_ok_rate
entropy
mean_probability_true_label
```

## Outputs

```text
data/judges/inferability_multi_judge.parquet
outputs/metrics/inferability_judge_stability.json
```

## Verification commands

```bash
python src/judges/run_multi_judge_inferability.py \
  --rationales data/rationales/parsed/train_candidates_strict.parquet \
  --config configs/default_paths.yaml \
  --output data/judges/inferability_multi_judge.parquet \
  --metrics outputs/metrics/inferability_judge_stability.json
```

## Acceptance criteria

- No parse failure defaults to neutral.
- At least one judge runs successfully.
- Label-order consistency is computed.
- Judge outputs include model name and prompt variant.


## Status JSON contract

When this task is complete, write:

```json
{
  "step": "10_PROXY_JUDGES_MULTI_MODEL_DEBIASED",
  "status": "PASS|FAIL",
  "inputs_checked": [],
  "outputs_created": [],
  "metrics": {},
  "failures": [],
  "next_step_allowed": true
}
```

Save it to:

```text
outputs/status/10_PROXY_JUDGES_MULTI_MODEL_DEBIASED.status.json
```

`next_step_allowed` must be `false` if any acceptance criterion fails.
