# 19 — Ablations and statistical tests

## Goal

Run real ablations. Do not generate placeholder or dummy ablation rows.

## Required ablations

```text
A0 Full FIRE-Fin
A1 No technical indicators
A2 No news
A3 No technical event tokens, raw numbers only
A4 No flow reward, proxy average only
A5 No grounding reward
A6 No regime-conditioned noise
A7 SFT only, no DPO
A8 No counterfactual pairs
```

## Codex task

Create:

```text
src/eval/run_ablation_suite.py
src/eval/statistical_tests.py
```

## Statistical tests

For prediction:
- paired bootstrap on test samples for macro-F1 and MCC.

For backtest:
- block bootstrap on daily portfolio returns.

Output confidence intervals:

```text
mean
std
95% CI
p-value vs full model if applicable
```

## Outputs

```text
outputs/tables/ablation_suite_results.csv
outputs/metrics/statistical_tests.json
```

## Verification commands

```bash
python src/eval/run_ablation_suite.py --config configs/ablations.yaml
python src/eval/statistical_tests.py \
  --predictions outputs/predictions/ \
  --daily-returns outputs/tables/backtest_daily_returns_v2.csv \
  --output outputs/metrics/statistical_tests.json
```

## Acceptance criteria

- No dummy rows.
- If an ablation is not run, table must say `NOT_RUN` and paper tables must not present it as evidence.
- Statistical tests use only test predictions and daily portfolio returns.
- Full model must be compared against at least proxy-average and SFT-only baselines.


## Status JSON contract

When this task is complete, write:

```json
{
  "step": "19_ABLATION_AND_STATISTICAL_TESTS",
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
outputs/status/19_ABLATION_AND_STATISTICAL_TESTS.status.json
```

`next_step_allowed` must be `false` if any acceptance criterion fails.
