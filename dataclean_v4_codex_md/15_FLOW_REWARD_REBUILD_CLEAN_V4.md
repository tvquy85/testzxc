# 15 — Flow Reward Rebuild on Clean V4 Data

> Scope: Upgrade `tvquy85/testzxc` branch `currentdata-aaai-fix-v2` on the already-built current dataset. Do not use SN2. Do not expand to full FNSPID. Do not overwrite v3 artifacts; all new artifacts must use `_v4` or `current_clean_v4`.
> Codex rule: implement only the task in this file, run verification, write status JSON, and never PASS if required outputs are missing or empty.

## Goal
Rebuild flow reward with cleaner, evidence-grounded targets. Do not claim improvement unless it beats proxy.

## Inputs
- `data/rationales/parsed/current_clean_train_qwen3_4b_v4.parquet`
- `data/judges/independent_inferability_evidence_v4.parquet`
- `data/judges/claim_grounding_evidence_v4.parquet`
- `data/processed/current_track_assignments_v4.parquet`

## Outputs
- `data/reward/flow_v4_dataset.pt`
- `checkpoints/reward/flow_v4/`
- `outputs/metrics/flow_vs_proxy_clean_v4.json`
- `outputs/status/15_FLOW_REWARD_REBUILD_CLEAN_V4.status.json`

## Target vector
`[true_label_probability_independent, inferability_confidence, news_grounding_score, technical_grounding_score, evidence_quality_weight, counterfactual_proxy_if_available, utility_proxy]`

## Conditioning priority
1. `all-MiniLM-L6-v2` embedding of evidence context + rationale.
2. FinBERT embedding.
3. hash fallback only if local embeddings unavailable; record `semantic_backend=false`.

## Commands
```bash
python -m src.reward.build_flow_dataset_v4 --rationales data/rationales/parsed/current_clean_train_qwen3_4b_v4.parquet --inferability data/judges/independent_inferability_evidence_v4.parquet --grounding data/judges/claim_grounding_evidence_v4.parquet --tracks data/processed/current_track_assignments_v4.parquet --output data/reward/flow_v4_dataset.pt --metrics outputs/metrics/flow_v4_dataset.json
python -m src.reward.train_flow_reward_v4 --dataset data/reward/flow_v4_dataset.pt --output-dir checkpoints/reward/flow_v4 --metrics outputs/metrics/flow_train_v4.json
python -m src.reward.evaluate_flow_vs_proxy_v4 --dataset data/reward/flow_v4_dataset.pt --checkpoint checkpoints/reward/flow_v4 --output outputs/metrics/flow_vs_proxy_clean_v4.json --status outputs/status/15_FLOW_REWARD_REBUILD_CLEAN_V4.status.json
```

## Claim gate
Allow `flow_reward_improvement=true` only if flow beats proxy in at least 2/3: rank corr, pair accuracy, top-decile utility.
