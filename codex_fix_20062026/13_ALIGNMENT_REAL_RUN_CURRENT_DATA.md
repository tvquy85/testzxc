# 13 — Real Current-Data Alignment

> Scope: upgrade `tvquy85/testzxc` branch `upgrade-aaai-reproducibility` on the **current already-built dataset/artifacts first**. Do not expand to full FNSPID until the current-data gates pass.
> Codex rule: implement only this task, run verification, write/update status JSON, and do not silently PASS if required outputs are missing.


## Goal
Run real alignment, not only 10-step smoke.

## Inputs
- `data/alignment/rwsft_current_v3.jsonl`
- `data/alignment/dpo_current_v3.jsonl`

## Outputs
- `checkpoints/aligned/qwen3_4b/current_v3_rwsft/`
- `checkpoints/aligned/qwen3_4b/current_v3_dpo/`
- `outputs/metrics/alignment_training_current_v3.json`
- `outputs/status/13_ALIGNMENT_REAL_RUN_CURRENT_DATA.status.json`

## Training target for RTX 3090
- QLoRA 4-bit if available
- max_seq_len 2048
- batch_size 1
- grad_accum 16
- RWSFT steps 300–800
- DPO steps 300–800

## Codex task
Create or adapt `src/alignment/train_current_v3.py`. Status FAIL if effective steps < 300 unless explicit OOM diagnostic is saved.

## Run
```bash
python -m src.alignment.train_current_v3 \
  --model qwen3_4b \
  --rwsft data/alignment/rwsft_current_v3.jsonl \
  --dpo data/alignment/dpo_current_v3.jsonl \
  --output-root checkpoints/aligned/qwen3_4b \
  --rwsft-max-steps 500 \
  --dpo-max-steps 500 \
  --max-seq-len 2048 \
  --batch-size 1 \
  --grad-accum 16 \
  --metrics outputs/metrics/alignment_training_current_v3.json \
  --status outputs/status/13_ALIGNMENT_REAL_RUN_CURRENT_DATA.status.json
```

## Verify
```bash
python - <<'PY'
import json, pathlib
m=json.load(open("outputs/metrics/alignment_training_current_v3.json"))
assert m["rwsft_steps"] >= 300
assert m["dpo_steps"] >= 300
assert pathlib.Path("checkpoints/aligned/qwen3_4b/current_v3_dpo").exists()
print(m)
PY
```

## Acceptance
- Real adapters saved.
- Loss curves saved.
- No performance claim until eval passes.
