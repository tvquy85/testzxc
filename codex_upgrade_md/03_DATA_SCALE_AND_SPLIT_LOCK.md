# 03 — Data scale and locked train/val/test split

## Goal

Move from a small 36-ticker prototype toward an AAAI-grade reproducible dataset split.

## Required data

Use FNSPID if locally available. If not, the script must print exact download instructions and stop safely.

Expected local locations:

```text
data/raw/fnspid/
data/raw/fnspid/news/
data/raw/fnspid/prices/
```

## Codex task

Create `src/data/build_dataset_manifest.py` and `src/data/build_locked_splits.py`.

### `build_dataset_manifest.py`

It must:
- scan raw FNSPID news and price files;
- detect columns automatically;
- report ticker count, row count, date min/max, missing values;
- save a manifest.

Output:

```text
outputs/manifests/fnspid_dataset_manifest.json
```

### `build_locked_splits.py`

It must:
- use chronological splits only;
- default split:
  - train: date <= 2021-12-31
  - validation: 2022-01-01 to 2022-12-31
  - test: date >= 2023-01-01
- never random split;
- write split membership by `sample_id`.

Output:

```text
data/processed/split_membership.parquet
outputs/manifests/split_manifest.json
```

## Minimum scale gates

For MVP pass:

```text
tickers >= 36
aligned candidate rows >= 100000
```

For AAAI target pass:

```text
tickers >= 300
aligned candidate rows >= 500000
```

Write both `mvp_gate` and `aaai_target_gate` in status JSON. Do not fail MVP if AAAI target is not met; mark it as `target_not_met`.

## Verification commands

```bash
python src/data/build_dataset_manifest.py --raw-root data/raw/fnspid --output outputs/manifests/fnspid_dataset_manifest.json
python src/data/build_locked_splits.py --input data/labels/aligned_samples_h1.parquet --output data/processed/split_membership.parquet
python - <<'PY'
import pandas as pd
s = pd.read_parquet("data/processed/split_membership.parquet")
print(s["split"].value_counts())
assert set(s["split"]).issubset({"train","val","test"})
PY
```

## Acceptance criteria

- Split membership is deterministic.
- No `sample_id` appears in more than one split.
- Date ranges do not overlap.
- Status records actual ticker count and rows.


## Status JSON contract

When this task is complete, write:

```json
{
  "step": "03_DATA_SCALE_AND_SPLIT_LOCK",
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
outputs/status/03_DATA_SCALE_AND_SPLIT_LOCK.status.json
```

`next_step_allowed` must be `false` if any acceptance criterion fails.
