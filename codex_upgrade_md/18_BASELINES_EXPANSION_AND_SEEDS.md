# 18 — Baseline expansion and multi-seed evaluation

## Goal

Strengthen empirical rigor with required baselines and seed variance.

## Baselines required

Implement or confirm:

```text
B1 FinBERT + Logistic Regression
B2 Technical LightGBM
B3 News + Technical Late Fusion
B4 DLinear or equivalent time-series baseline
B5 Chronos-Bolt-small price-only
B6 IBM TTM-r2 price-only
B7 MOMENT embedding + classifier
B8 LLM zero-shot
B9 LLM SFT only
B10 LLM RWSFT only
B11 LLM DPO scalar proxy reward
B12 FIRE-Fin flow v2
```

## Codex task

Create:

```text
src/eval/run_baseline_suite.py
src/eval/aggregate_seed_results.py
configs/baselines.yaml
```

Use seeds:

```text
[11, 22, 33]
```

For GPU-heavy LLM baselines, allow one seed for MVP but status must mark `multi_seed_missing=true`.

## Outputs

```text
outputs/metrics/baseline_suite_summary.json
outputs/tables/baseline_suite_by_seed.csv
outputs/tables/baseline_suite_aggregate.csv
```

## Verification commands

```bash
python src/eval/run_baseline_suite.py --config configs/baselines.yaml --seeds 11 22 33
python src/eval/aggregate_seed_results.py \
  --input outputs/tables/baseline_suite_by_seed.csv \
  --output outputs/tables/baseline_suite_aggregate.csv
```

## Acceptance criteria

- At least B1–B4 run for all 3 seeds.
- Missing baselines are explicit, not omitted.
- Mean ± std is reported.
- No test-set tuning.


## Status JSON contract

When this task is complete, write:

```json
{
  "step": "18_BASELINES_EXPANSION_AND_SEEDS",
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
outputs/status/18_BASELINES_EXPANSION_AND_SEEDS.status.json
```

`next_step_allowed` must be `false` if any acceptance criterion fails.
