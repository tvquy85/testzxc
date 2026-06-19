# 20 — Paper tables with no dummy values

## Goal

Replace `build_paper_tables.py` behavior that creates dummy tables with zero values.

## Existing file to replace

```text
src/eval/build_paper_tables.py
```

Create:

```text
src/eval/build_paper_tables_v2.py
```

## Required behavior

1. Read only real metrics from `outputs/metrics` and `outputs/tables`.
2. If required metrics are missing, fail loudly.
3. Do not create zero-filled placeholder rows.
4. Every table must include:
   - source metric file;
   - run id;
   - seed;
   - split;
   - timestamp.

## Tables

```text
table_1_prediction_main.csv
table_2_explanation_quality.csv
table_3_daily_portfolio_backtest.csv
table_4_counterfactual_directional.csv
table_5_ablation.csv
table_6_scale_and_compute.csv
```

## Verification commands

```bash
python src/eval/build_paper_tables_v2.py \
  --metrics-root outputs/metrics \
  --tables-root outputs/tables \
  --output-root outputs/tables/final
```

## Acceptance criteria

- No value is inserted as dummy zero.
- Missing required metric causes non-zero exit.
- `outputs/tables/final/table_manifest.json` maps every cell/table to source files.
- Old `build_paper_tables.py` is not used in final package.


## Status JSON contract

When this task is complete, write:

```json
{
  "step": "20_PAPER_TABLES_NO_DUMMY_GATES",
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
outputs/status/20_PAPER_TABLES_NO_DUMMY_GATES.status.json
```

`next_step_allowed` must be `false` if any acceptance criterion fails.
