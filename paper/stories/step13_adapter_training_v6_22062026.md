# Step 13 Story - Real V6 Adapters Trained, Claims Still Deferred

Date: 2026-06-22

Step 13 moved the pipeline from alignment data to real trainable artifacts. The important story is that adapter existence is now verified, but downstream claims are still blocked until prediction evaluation.

Training was staged:

```text
RWSFT smoke: 2 steps, PASS
DPO smoke: 2 steps, PASS
RWSFT final: 800 steps, PASS
DPO final: 800 steps, PASS
summary gate: PASS
```

The environment gate was clean. The run used the existing `D:\LOBProj\LOBExp\.venv` environment, local Qwen3 cache under `E:\huggingface`, and the RTX 3090. No model download was needed. Required training modules were available: `torch`, `transformers`, `peft`, `trl`, and `bitsandbytes`.

Final artifacts:

```text
outputs/models/qwen3_current_v6_rwsft_adapter/adapter_model.safetensors
outputs/models/qwen3_current_v6_dpo_adapter/adapter_model.safetensors
outputs/metrics/13_v6_alignment_training.json
outputs/status/13_TRAIN_RWSFT_DPO_V6.status.json
```

Key metrics:

```text
RWSFT loss: 2.1356 -> 0.4755
RWSFT records loaded: 1600
DPO loss: 0.6942 -> 0.7155
DPO records loaded: 316
min_steps gate: 800 for both adapters
full tests: 68 passed
```

Paper angle to preserve: the adapter training is a necessary engineering milestone, not an evidence claim. RWSFT loss decreased strongly, so the supervised alignment path is technically healthy. DPO produced a valid adapter but its training loss did not improve; that should be reported honestly and then judged by Step 14 prediction metrics rather than by training loss alone.

Potential wording:

```text
We trained V6-specific RWSFT and DPO adapters from the current-data alignment set rather than reusing earlier V5 artifacts. The training gate verifies that both adapters exist after 800 QLoRA steps. However, the DPO loss did not decrease, so we treat adapter training as an artifact gate only and defer all performance claims to held-out prediction and baseline comparisons.
```
