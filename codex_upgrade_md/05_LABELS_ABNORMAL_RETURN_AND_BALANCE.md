# 05 — Abnormal-return labels and class balance audit

## Goal

Replace naive raw-return labeling with abnormal-return labeling and clear class-balance diagnostics.

## Files to create or modify

```text
src/data/build_abnormal_return_labels.py
src/eval/label_distribution_report.py
```

## Label rule

Default horizon: `h1`.

Use:

```text
abnormal_return = stock_forward_return - market_forward_return
```

If beta-adjusted market returns are available, support:

```text
abnormal_return = stock_forward_return - beta_i * market_forward_return
```

Use 5 classes:

```text
strong_down: <= -0.03
mild_down: > -0.03 and <= -0.0075
neutral: > -0.0075 and < 0.0075
mild_up: >= 0.0075 and < 0.03
strong_up: >= 0.03
```

Also support quantile labels via `--label-mode quantile`.

## Codex task

1. Implement the labeling script.
2. Save both continuous abnormal return and class label.
3. Join locked split membership.
4. Write label-distribution reports by:
   - total;
   - split;
   - year;
   - ticker;
   - regime.

## Outputs

```text
data/labels/labels_h1_abnormal.parquet
outputs/metrics/label_distribution_h1.json
outputs/tables/label_distribution_by_split.csv
```

## Verification commands

```bash
python src/data/build_abnormal_return_labels.py --horizon 1 --output data/labels/labels_h1_abnormal.parquet
python src/eval/label_distribution_report.py --labels data/labels/labels_h1_abnormal.parquet --output outputs/metrics/label_distribution_h1.json
```

## Acceptance criteria

- `label_5` has exactly 5 legal values.
- No missing abnormal returns in final labeled set.
- Split-level class distribution is reported.
- Minority classes are not silently dropped.


## Status JSON contract

When this task is complete, write:

```json
{
  "step": "05_LABELS_ABNORMAL_RETURN_AND_BALANCE",
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
outputs/status/05_LABELS_ABNORMAL_RETURN_AND_BALANCE.status.json
```

`next_step_allowed` must be `false` if any acceptance criterion fails.
