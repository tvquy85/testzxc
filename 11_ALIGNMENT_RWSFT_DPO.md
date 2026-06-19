# Step 11 — Alignment with Reward-Weighted SFT and DPO

> **Use with Antigravity / Gemini Pro 3.1 High**  
> Treat this file as one bounded task. Do not implement later steps unless this file explicitly asks for them.  
> Always create small scripts, run verification, and save a short status JSON before stopping.

## Goal
Create preference pairs from flow rewards and fine-tune the main Explanation LLM using QLoRA. Start with reward-weighted SFT; run DPO only after SFT works.

## Inputs

```text
data/rationales/candidate_rationales_h1.jsonl
data/judge_outputs/flow_rewards_h1.parquet
data/labels/aligned_samples_h1.parquet
data/indicators/technical_event_tokens_h1.parquet
prompts/rationale_generation_prompt.txt
configs/local_paths.yaml
```

## Outputs

```text
data/rationales/preference_pairs_h1.jsonl
data/rationales/rwsft_train_h1.jsonl
outputs/checkpoints/rwsft_qwen_h1/
outputs/checkpoints/dpo_qwen_h1/
outputs/metrics/alignment_metrics_h1.json
outputs/status/11_ALIGNMENT_RWSFT_DPO.status.json
```

## Model priority

1. `Qwen3-4B-Instruct-2507`
2. `Qwen2.5-3B-Instruct`

Use QLoRA 4-bit.

## Tasks

### 1. Build preference pairs
Create `src/align/build_preference_pairs.py`. For each sample with at least 2 candidates:

- preferred = highest `flow_overall_reward`
- rejected = lowest `flow_overall_reward`
- require margin >= 0.10

### 2. Build RWSFT data
Create `src/align/build_rwsft_data.py`. Use top-1 or top-2 candidates per sample with weight = flow reward.

### 3. Train RWSFT
Create `src/align/train_rwsft_qlora.py`.

RTX 3090 config:

```yaml
load_in_4bit: true
bnb_4bit_quant_type: nf4
max_seq_length: 2048
per_device_train_batch_size: 1
gradient_accumulation_steps: 16
learning_rate: 2e-5
num_train_epochs: 1
lora_r: 16
lora_alpha: 32
lora_dropout: 0.05
fp16: true
gradient_checkpointing: true
```

### 4. Train DPO only if RWSFT passes
Create `src/align/train_dpo_qlora.py`. If DPO fails due to VRAM, stop after RWSFT and mark DPO as deferred.

## Verification
Run:

```bash
cd firefin
python src/align/build_preference_pairs.py \
  --rationales data/rationales/candidate_rationales_h1.jsonl \
  --flow-rewards data/judge_outputs/flow_rewards_h1.parquet \
  --output data/rationales/preference_pairs_h1.jsonl \
  --min-margin 0.10

python src/align/build_rwsft_data.py \
  --rationales data/rationales/candidate_rationales_h1.jsonl \
  --flow-rewards data/judge_outputs/flow_rewards_h1.parquet \
  --output data/rationales/rwsft_train_h1.jsonl

python src/align/train_rwsft_qlora.py \
  --train data/rationales/rwsft_train_h1.jsonl \
  --config configs/local_paths.yaml \
  --limit 128 \
  --output outputs/checkpoints/rwsft_qwen_h1_dryrun
```

## Acceptance criteria
PASS only if preference pairs >= 500, RWSFT data exists, and RWSFT dry-run saves adapter.

## Status JSON

```json
{
  "step": "11_ALIGNMENT_RWSFT_DPO",
  "status": "PASS|FAIL",
  "preference_pair_count": 0,
  "rwsft_rows": 0,
  "rwsft_checkpoint": "...",
  "dpo_checkpoint": "...|deferred",
  "gpu_memory_notes": "..."
}
```
