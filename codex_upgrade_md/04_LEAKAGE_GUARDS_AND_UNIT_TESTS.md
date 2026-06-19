# 04 — Leakage guards and unit tests

## Goal

Prevent future information from entering training, reward construction, prompts, or backtest.

## Files to create

```text
src/data/leakage_checks.py
tests/test_leakage_checks.py
```

## Codex task

Implement checks for:

1. **Timestamp rule**
   - news timestamp must be <= decision timestamp;
   - feature window end date must be <= decision date;
   - future return window must start after decision date.

2. **Split rule**
   - training rationales only from train split;
   - judge outputs used for training only from train split;
   - validation only for model selection;
   - test only for final evaluation.

3. **Prompt leakage rule**
   - prompts must not include realized return, future close, future label, or text like `shares rose after`.

4. **Output leakage rule**
   - alignment datasets must contain `split=train` only.

## Required script behavior

`src/data/leakage_checks.py` must accept:

```bash
python src/data/leakage_checks.py \
  --samples data/labels/aligned_samples_h1.parquet \
  --splits data/processed/split_membership.parquet \
  --output outputs/audit/leakage_report.json
```

## Verification commands

```bash
pytest -q tests/test_leakage_checks.py
python src/data/leakage_checks.py --samples data/labels/aligned_samples_h1.parquet --splits data/processed/split_membership.parquet --output outputs/audit/leakage_report.json
```

## Acceptance criteria

- Unit tests include at least one synthetic leak and verify it is caught.
- Leakage report includes `num_leaks`, `leak_examples`, and `passed`.
- `passed` must be false if any hard leak is found.


## Status JSON contract

When this task is complete, write:

```json
{
  "step": "04_LEAKAGE_GUARDS_AND_UNIT_TESTS",
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
outputs/status/04_LEAKAGE_GUARDS_AND_UNIT_TESTS.status.json
```

`next_step_allowed` must be `false` if any acceptance criterion fails.
