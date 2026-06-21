# 14 — Alignment Train Adapter V4 Medium

> **Scope:** work on branch `currentdata-aaai-fix-v2`, current-data only. Do **not** use SN2. Do **not** expand to full FNSPID. Build on existing DataClean V4 artifacts and write new artifacts with suffix `_medium_v5` or `_clean_v4_medium`.  
> **Codex rule:** execute one file at a time, verify, write status JSON, stop on FAIL. PASS means the step ran; it does not automatically allow a scientific claim.


## Goal
Train a real medium adapter: RWSFT then DPO.

## Why this is needed
The report stated V4 adapter was not trained/evaluated at sufficient scale. Medium backtest must not use old `current_v3_dpo`.

## Files to create or modify
- Modify only the files explicitly named in the command section. If a required file is missing, create it under the stated path.

## Inputs
```text
data/alignment/medium_clean_v4_rwsft.jsonl
data/alignment/medium_clean_v4_dpo_pairs.jsonl
configs/default_paths.yaml
```

## Outputs
```text
outputs/models/qwen3_medium_clean_v4_rwsft_adapter/
outputs/models/qwen3_medium_clean_v4_dpo_adapter/
outputs/metrics/14_alignment_train_medium.json
outputs/status/14_ALIGNMENT_TRAIN_ADAPTER_V4_MEDIUM.status.json
```

## Commands
```bash
python -m src.alignment.train_rwsft_qlora \
  --model-key qwen3_4b --config configs/default_paths.yaml \
  --train-file data/alignment/medium_clean_v4_rwsft.jsonl \
  --output-dir outputs/models/qwen3_medium_clean_v4_rwsft_adapter \
  --max-steps 800 --batch-size 1 --grad-accum 16 --lr 2e-5 --max-seq-len 2048 \
  --save-steps 200 \
  --metrics outputs/metrics/14_rwsft_train_medium.json \
  --status outputs/status/14_RWSFT_TRAIN_MEDIUM.status.json

python -m src.alignment.train_dpo_qlora \
  --model-key qwen3_4b \
  --base-adapter outputs/models/qwen3_medium_clean_v4_rwsft_adapter \
  --train-file data/alignment/medium_clean_v4_dpo_pairs.jsonl \
  --output-dir outputs/models/qwen3_medium_clean_v4_dpo_adapter \
  --max-steps 800 --batch-size 1 --grad-accum 16 --lr 5e-6 --max-seq-len 2048 \
  --save-steps 200 \
  --metrics outputs/metrics/14_dpo_train_medium.json \
  --status outputs/status/14_ALIGNMENT_TRAIN_ADAPTER_V4_MEDIUM.status.json
```

## Verification
```bash
python - <<'PY'
from pathlib import Path
assert list(Path('outputs/models/qwen3_medium_clean_v4_rwsft_adapter').glob('**/adapter_model*'))
assert list(Path('outputs/models/qwen3_medium_clean_v4_dpo_adapter').glob('**/adapter_model*'))
print('PASS adapter exists')
PY
```

## Acceptance criteria
- Real adapter files exist.
- Not a smoke run: >= 500 optimizer steps each unless OOM is explicitly documented with FAIL.
- If OOM, do not silently PASS/defer.

## Status JSON contract
Write a status file with this shape:
```json
{
  "step": "14_ALIGNMENT_TRAIN_ADAPTER_V4_MEDIUM",
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
