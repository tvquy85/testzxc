# 13 — Train RWSFT and DPO V6 Adapters

> Scope: branch `currentdata-aaai-fix-v2`; current-data only; do not use SN2; do not expand full FNSPID. Execute one file at a time. Every step must write a status JSON and must stop on FAIL. Pipeline PASS is not a scientific claim.

## Goal
Train real V6 adapters. Do not use V5 adapters for V6 prediction.

## Outputs
```text
outputs/models/qwen3_current_v6_rwsft_adapter/adapter_model.safetensors
outputs/models/qwen3_current_v6_dpo_adapter/adapter_model.safetensors
outputs/metrics/13_v6_alignment_training.json
outputs/status/13_TRAIN_RWSFT_DPO_V6.status.json
```

## Training config
Qwen3-4B-Instruct-2507, QLoRA/4-bit if supported, LoRA r=16, max_seq_len=2048, batch_size=1, grad_accum=16, fp16, max_steps=800 minimum.

## Test case
```python
from pathlib import Path

def test_v6_adapters_exist():
    assert Path('outputs/models/qwen3_current_v6_rwsft_adapter/adapter_model.safetensors').exists()
    assert Path('outputs/models/qwen3_current_v6_dpo_adapter/adapter_model.safetensors').exists()
```

## Commands
```bash
python -m src.alignment.train_rwsft_qlora --train-jsonl data/alignment/current_v6_rwsft.jsonl --config configs/local_paths.yaml --output-dir outputs/models/qwen3_current_v6_rwsft_adapter --max-steps 800 --status outputs/status/13A_TRAIN_RWSFT_V6.status.json
python -m src.alignment.train_dpo_qlora --train-jsonl data/alignment/current_v6_dpo_pairs.jsonl --config configs/local_paths.yaml --output-dir outputs/models/qwen3_current_v6_dpo_adapter --max-steps 800 --status outputs/status/13B_TRAIN_DPO_V6.status.json
python -m src.alignment.summarize_training_v6 --rwsft outputs/status/13A_TRAIN_RWSFT_V6.status.json --dpo outputs/status/13B_TRAIN_DPO_V6.status.json --metrics outputs/metrics/13_v6_alignment_training.json --status outputs/status/13_TRAIN_RWSFT_DPO_V6.status.json
python -m pytest -q tests/test_v6_adapter_files.py tests
```

## Progress Update 2026-06-22
Status: `stage_3_alignment_training PASS`; both V6 adapters were trained from the Step 12 current-data alignment artifacts. Status file reports `PASS` and `next_step_allowed=true`.

Artifacts verified:
```text
outputs/models/qwen3_current_v6_rwsft_adapter/adapter_model.safetensors
outputs/models/qwen3_current_v6_dpo_adapter/adapter_model.safetensors
outputs/metrics/13A_v6_rwsft_train.json
outputs/metrics/13B_v6_dpo_train.json
outputs/metrics/13_v6_alignment_training.json
outputs/manifests/13A_TRAIN_RWSFT_V6.manifest.json
outputs/manifests/13B_TRAIN_DPO_V6.manifest.json
outputs/manifests/13_TRAIN_RWSFT_DPO_V6.manifest.json
outputs/status/13A_TRAIN_RWSFT_V6.status.json
outputs/status/13B_TRAIN_DPO_V6.status.json
outputs/status/13_TRAIN_RWSFT_DPO_V6.status.json
```

Training sequence:
```text
preflight: PASS, RTX 3090 available, local Qwen3 cache available, torch/transformers/peft/trl/bitsandbytes import OK
RWSFT smoke: PASS, 2 steps, output dir outputs/models/qwen3_current_v6_rwsft_adapter_smoke
DPO smoke: PASS, 2 steps, output dir outputs/models/qwen3_current_v6_dpo_adapter_smoke
RWSFT final: PASS, 800 steps
DPO final: PASS, 800 steps, initialized from final RWSFT adapter
summary: PASS, min_steps=800
```

Gate metrics from `outputs/metrics/13_v6_alignment_training.json`:
```text
rwsft.max_steps: 800
rwsft.adapter_exists: true
rwsft.loss_first: 2.1356
rwsft.loss_last: 0.4755
rwsft.train_records_loaded: 1600
rwsft.max_memory_allocated_bytes: 8001580032

dpo.max_steps: 800
dpo.adapter_exists: true
dpo.loss_first: 0.6942
dpo.loss_last: 0.7155
dpo.train_records_loaded: 316
dpo.max_memory_allocated_bytes: 8556736512

adapters_exist: true
claim_allowed: false
```

Acceptance result:
```text
RWSFT adapter gate: PASS
DPO adapter gate: PASS
minimum training steps gate: PASS (800 >= 800 for both)
status JSON gate: PASS
adapter file test: PASS
full tests: PASS (68 passed, 2 warnings)
GPU cleanup: PASS (0 MiB used after completion)
```

Verification commands run:
```bash
D:\LOBProj\LOBExp\.venv\Scripts\python.exe -m src.alignment.train_rwsft_qlora --train-jsonl data/alignment/current_v6_rwsft.jsonl --config configs/local_paths.yaml --output-dir outputs/models/qwen3_current_v6_rwsft_adapter --max-steps 800 --batch-size 1 --max-seq-len 2048 --metrics outputs/metrics/13A_v6_rwsft_train.json --status outputs/status/13A_TRAIN_RWSFT_V6.status.json --manifest outputs/manifests/13A_TRAIN_RWSFT_V6.manifest.json
D:\LOBProj\LOBExp\.venv\Scripts\python.exe -m src.utils.verify_status --status outputs/status/13A_TRAIN_RWSFT_V6.status.json
D:\LOBProj\LOBExp\.venv\Scripts\python.exe -m src.alignment.train_dpo_qlora --train-jsonl data/alignment/current_v6_dpo_pairs.jsonl --base-adapter outputs/models/qwen3_current_v6_rwsft_adapter --config configs/local_paths.yaml --output-dir outputs/models/qwen3_current_v6_dpo_adapter --max-steps 800 --batch-size 1 --max-seq-len 2048 --metrics outputs/metrics/13B_v6_dpo_train.json --status outputs/status/13B_TRAIN_DPO_V6.status.json --manifest outputs/manifests/13B_TRAIN_DPO_V6.manifest.json
D:\LOBProj\LOBExp\.venv\Scripts\python.exe -m src.utils.verify_status --status outputs/status/13B_TRAIN_DPO_V6.status.json
D:\LOBProj\LOBExp\.venv\Scripts\python.exe -m src.alignment.summarize_training_v6 --rwsft outputs/status/13A_TRAIN_RWSFT_V6.status.json --dpo outputs/status/13B_TRAIN_DPO_V6.status.json --metrics outputs/metrics/13_v6_alignment_training.json --status outputs/status/13_TRAIN_RWSFT_DPO_V6.status.json --manifest outputs/manifests/13_TRAIN_RWSFT_DPO_V6.manifest.json --min-steps 800
D:\LOBProj\LOBExp\.venv\Scripts\python.exe -m src.utils.verify_status --status outputs/status/13_TRAIN_RWSFT_DPO_V6.status.json
D:\LOBProj\LOBExp\.venv\Scripts\python.exe -m pytest -q tests
```

Implementation notes:
```text
src/alignment/train_rwsft_qlora.py and train_dpo_qlora.py now accept --train-jsonl as an alias for --train-file, matching this runbook.
src/alignment/summarize_training_v6.py enforces child status PASS, adapter existence, finite losses, and max_steps >= 800 before the Step 13 summary can PASS.
```

Scientific caution: Step 13 proves that real V6 adapters exist and were trained for the required minimum steps. It does not prove the adapters improve forecasts. DPO final loss did not decrease (`0.6942 -> 0.7155`), so Step 14 must evaluate base/SFT/RWSFT/DPO prediction quality before any alignment or forecast improvement claim is allowed.
