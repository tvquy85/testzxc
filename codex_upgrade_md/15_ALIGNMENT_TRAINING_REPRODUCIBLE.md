# 15 — Reproducible RWSFT and DPO training

## Goal

Train aligned LLMs reproducibly without silently skipping training.

## Existing issue

Any script that catches OOM and exits with status PASS or `exit 0` while training is deferred must be fixed.

## Files to inspect or modify

```text
src/alignment/train_rwsft*.py
src/alignment/train_dpo*.py
```

## Model plan for RTX 3090

Use 4-bit QLoRA:

```text
main: Qwen3-4B-Instruct-2507
fallback: DeepSeek-R1-Distill-Qwen-1.5B
```

Recommended settings:

```text
load_in_4bit=True
bnb_4bit_quant_type=nf4
gradient_checkpointing=True
per_device_train_batch_size=1
gradient_accumulation_steps=16
max_seq_length=2048
lora_r=16
lora_alpha=32
```

## Codex task

1. Add explicit `--dry-run` mode.
2. Add explicit `--max-steps` for smoke test.
3. On OOM, write `status=FAIL`, not PASS.
4. Save training config, seed, dataset hash, model path, LoRA config.
5. Save adapter to:

```text
checkpoints/aligned/<model_name>/rwsft_v2/
checkpoints/aligned/<model_name>/dpo_v2/
```

## Verification commands

```bash
python src/alignment/train_rwsft_v2.py \
  --train data/alignment/rwsft_train_v2.jsonl \
  --model qwen3_4b \
  --output-dir checkpoints/aligned/qwen3_4b/rwsft_v2 \
  --max-steps 10

python src/alignment/train_dpo_v2.py \
  --train data/alignment/dpo_pairs_train_v2.jsonl \
  --model qwen3_4b \
  --output-dir checkpoints/aligned/qwen3_4b/dpo_v2 \
  --max-steps 10
```

## Acceptance criteria

- Smoke training produces adapter files.
- Status FAIL on OOM or missing model.
- No deferred training is marked PASS.
- Dataset hash is saved in trainer state.


## Status JSON contract

When this task is complete, write:

```json
{
  "step": "15_ALIGNMENT_TRAINING_REPRODUCIBLE",
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
outputs/status/15_ALIGNMENT_TRAINING_REPRODUCIBLE.status.json
```

`next_step_allowed` must be `false` if any acceptance criterion fails.
