# 14 — Independent Inferability Judge V4

> Scope: Upgrade `tvquy85/testzxc` branch `currentdata-aaai-fix-v2` on the already-built current dataset. Do not use SN2. Do not expand to full FNSPID. Do not overwrite v3 artifacts; all new artifacts must use `_v4` or `current_clean_v4`.
> Codex rule: implement only the task in this file, run verification, write status JSON, and never PASS if required outputs are missing or empty.

## Goal
Rerun inferability using an independent judge that reads `context + rationale`, not the generator's self-declared forecast distribution.

## Inputs
- `data/rationales/parsed/current_clean_train_qwen3_4b_v4.parquet`
- `data/processed/ticker_date_evidence_contexts_h1_v4.parquet`

## Outputs
- `data/judges/independent_inferability_evidence_v4.parquet`
- `outputs/metrics/independent_inferability_evidence_v4.json`
- `outputs/status/14_INDEPENDENT_JUDGE_RERUN_EVIDENCE_V4.status.json`

## Command
```bash
python -m src.judges.independent_inferability_judge_v4 \
  --rationales data/rationales/parsed/current_clean_train_qwen3_4b_v4.parquet \
  --contexts data/processed/ticker_date_evidence_contexts_h1_v4.parquet \
  --output data/judges/independent_inferability_evidence_v4.parquet \
  --metrics outputs/metrics/independent_inferability_evidence_v4.json \
  --status outputs/status/14_INDEPENDENT_JUDGE_RERUN_EVIDENCE_V4.status.json \
  --num-samples 6000 \
  --temperature 0.0 \
  --randomize-label-order true
```

## Gates
- schema OK >= 0.95.
- report mean true-label probability and random baseline 0.20.
- report by track.
- if true-label probability <= 0.22, set `inferability_claim_allowed=false`.
