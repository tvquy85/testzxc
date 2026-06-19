# 17 — Directional counterfactual evaluation v2

## Goal

Replace raw flip-rate CFR with direction-aware counterfactual metrics.

## Existing files to inspect

```text
src/eval/build_counterfactual_contexts.py
src/eval/evaluate_counterfactual_consistency.py
```

## Codex task

Create:

```text
src/eval/build_counterfactual_contexts_v2.py
src/eval/evaluate_counterfactual_directional_v2.py
```

## Required counterfactuals

1. Remove bad news.
   - Expected: `P(down)` decreases.
2. Remove good news.
   - Expected: `P(up)` decreases.
3. Neutralize overbought/oversold technical tokens.
   - Expected: reversal-risk confidence changes in the correct direction.
4. Flip high-volatility to normal-volatility.
   - Expected: entropy/confidence changes, not necessarily label flip.

## Metric

For each counterfactual, compute:

```text
delta_expected = signed change in expected direction
directional_pass = delta_expected > min_delta
```

Also compute:

```text
mean_delta
pass_rate
wrong_direction_rate
no_change_rate
```

## Generation settings

Use deterministic generation for evaluation:

```text
temperature=0.0
do_sample=False
```

Run multiple seeds only if stochastic mode is explicitly requested.

## Outputs

```text
outputs/metrics/counterfactual_directional_v2.json
outputs/tables/counterfactual_directional_examples.csv
```

## Verification commands

```bash
python src/eval/build_counterfactual_contexts_v2.py \
  --samples data/labels/labels_h1_abnormal.parquet \
  --tokens data/indicators/technical_event_tokens_h1_v2.parquet \
  --split test \
  --output data/processed/counterfactual_tasks_v2.jsonl

python src/eval/evaluate_counterfactual_directional_v2.py \
  --checkpoint checkpoints/aligned/qwen3_4b/dpo_v2 \
  --input data/processed/counterfactual_tasks_v2.jsonl \
  --output outputs/metrics/counterfactual_directional_v2.json
```

## Acceptance criteria

- Metric is directional, not only flip/no-flip.
- Examples include original and counterfactual distributions.
- Evaluation uses test split only.
- Prompt context is not empty in the judge.


## Status JSON contract

When this task is complete, write:

```json
{
  "step": "17_COUNTERFACTUAL_DIRECTIONAL_EVAL_V2",
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
outputs/status/17_COUNTERFACTUAL_DIRECTIONAL_EVAL_V2.status.json
```

`next_step_allowed` must be `false` if any acceptance criterion fails.
