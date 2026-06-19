# 09 — Rationale generation scale-up, train-only

## Goal

Generate enough rationales for alignment while preventing test contamination.

## Files to modify or create

```text
src/llm/generate_rationales.py
src/llm/build_rationale_prompts.py
```

## Model plan

Use local models only for reproducibility:

```text
primary generator: Qwen3-4B-Instruct-2507
backup generator: only an explicitly available local instruct model after cache verification
judge later: Llama-3-8B-Instruct
```

## Codex task

1. Add mandatory argument:

```bash
--split train
```

Generation must refuse to run on `val` or `test` unless `--allow-eval-generation` is explicitly set for evaluation-only output.

2. Generate multiple candidates per sample:

```bash
--num-candidates 4
```

3. Save raw outputs:

```text
data/rationales/raw/train_candidates.jsonl
```

4. Save strict parsed outputs:

```text
data/rationales/parsed/train_candidates_strict.parquet
```

5. Include these fields:

```text
sample_id
split
ticker
event_timestamp
candidate_id
model_name
prompt_hash
raw_text
parse_ok
parse_errors
parsed_json
```

## Verification commands

```bash
python src/llm/generate_rationales.py \
  --input data/labels/labels_h1_abnormal.parquet \
  --splits data/processed/split_membership.parquet \
  --split train \
  --num-samples 1000 \
  --num-candidates 4 \
  --model qwen3_4b \
  --raw-output data/rationales/raw/train_candidates.jsonl \
  --parsed-output data/rationales/parsed/train_candidates_strict.parquet

python - <<'PY'
import pandas as pd
df = pd.read_parquet("data/rationales/parsed/train_candidates_strict.parquet")
assert set(df["split"]) == train
print(df["parse_ok"].mean())
PY
```

## Acceptance criteria

- 100% of generated rows have `split=train`.
- Raw and parsed outputs have matching row counts.
- Parse-ok rate is reported, not fixed.
- Prompt hash is saved.


## Status JSON contract

When this task is complete, write:

```json
{
  "step": "09_RATIONALE_GENERATION_SCALEUP_TRAIN_ONLY",
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
outputs/status/09_RATIONALE_GENERATION_SCALEUP_TRAIN_ONLY.status.json
```

`next_step_allowed` must be `false` if any acceptance criterion fails.
