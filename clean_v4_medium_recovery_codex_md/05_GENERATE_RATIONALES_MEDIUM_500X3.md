# 05 — Generate Rationales Medium 500x3

> **Scope:** work on branch `currentdata-aaai-fix-v2`, current-data only. Do **not** use SN2. Do **not** expand to full FNSPID. Build on existing DataClean V4 artifacts and write new artifacts with suffix `_medium_v5` or `_clean_v4_medium`.  
> **Codex rule:** execute one file at a time, verify, write status JSON, stop on FAIL. PASS means the step ran; it does not automatically allow a scientific claim.


## Goal
Generate 500–1000 train sample IDs × 3 candidates with Qwen3-4B and the strict evidence-ID prompt.

## Why this is needed
Flow, DPO, and judge signals need multiple candidates per sample. V4 small-scale 100×3 is not enough.

## Files to create or modify
- Modify only the files explicitly named in the command section. If a required file is missing, create it under the stated path.

## Inputs
```text
data/processed/medium_clean_v4_contexts_gated.parquet
prompts/rationale_generation_prompt_evidence_v4.txt
configs/default_paths.yaml
```

## Outputs
```text
data/rationales/raw/medium_clean_v4_qwen3_4b_candidates.jsonl
data/rationales/parsed/medium_clean_v4_qwen3_4b_candidates.parquet
outputs/metrics/05_generate_rationales_medium.json
outputs/status/05_GENERATE_RATIONALES_MEDIUM_500X3.status.json
```

## Commands
```bash
python -m src.llm.generate_rationales \
  --input data/processed/medium_clean_v4_contexts_gated.parquet \
  --prompt prompts/rationale_generation_prompt_evidence_v4.txt \
  --config configs/default_paths.yaml \
  --split train \
  --model qwen3_4b \
  --num-samples 500 \
  --num-candidates 3 \
  --max-new-tokens 256 \
  --max-input-tokens 3072 \
  --temperature 0.45 \
  --top-p 0.88 \
  --top-k 40 \
  --batch-size 4 \
  --sort-by-length \
  --save-every 50 \
  --resume \
  --schema-version v4 \
  --stage medium_clean_v4 \
  --raw-output data/rationales/raw/medium_clean_v4_qwen3_4b_candidates.jsonl \
  --parsed-output data/rationales/parsed/medium_clean_v4_qwen3_4b_candidates.parquet \
  --metrics outputs/metrics/05_generate_rationales_medium.json \
  --status outputs/status/05_GENERATE_RATIONALES_MEDIUM_500X3.status.json
```

## Verification
```bash
python - <<'PY'
import pandas as pd, json
df=pd.read_parquet('data/rationales/parsed/medium_clean_v4_qwen3_4b_candidates.parquet')
assert df.sample_id.nunique() >= 500
assert len(df) >= 1000
m=json.load(open('outputs/metrics/05_generate_rationales_medium.json'))
assert m.get('parse_ok_rate',0) >= 0.95
assert m.get('schema_ok_rate',0) >= 0.95
print('PASS rationale medium', len(df))
PY
```

## Acceptance criteria
- `parse_ok_rate >= 0.95`.
- `schema_ok_rate >= 0.95`.
- At least 500 unique train sample IDs.
- At least two candidates/sample on average.

## Status JSON contract
Write a status file with this shape:
```json
{
  "step": "05_GENERATE_RATIONALES_MEDIUM_500X3",
  "status": "PASS|FAIL",
  "pipeline_pass": true,
  "claim_allowed": false,
  "inputs": [],
  "outputs": [],
  "metrics": {},
  "failures": [],
  "warnings": []
}
```
