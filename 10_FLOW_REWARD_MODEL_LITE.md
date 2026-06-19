# Step 10 — Lightweight Rectified Flow Reward Model

> **Use with Antigravity / Gemini Pro 3.1 High**  
> Treat this file as one bounded task. Do not implement later steps unless this file explicitly asks for them.  
> Always create small scripts, run verification, and save a short status JSON before stopping.

## Goal
Train a small conditional rectified-flow reward model that maps Gaussian noise to judge-derived reward distributions.

## Inputs

```text
data/judge_outputs/judge_scores_h1.parquet
data/rationales/candidate_rationales_h1.jsonl
data/indicators/technical_features_h1.parquet
configs/local_paths.yaml
```

## Outputs

```text
outputs/checkpoints/flow_reward_lite_h1.pt
outputs/metrics/flow_reward_lite_h1.json
data/judge_outputs/flow_rewards_h1.parquet
outputs/status/10_FLOW_REWARD_MODEL_LITE.status.json
```

## Method
Target vector:

```text
z1 = [infer_p_strong_down, infer_p_mild_down, infer_p_neutral, infer_p_mild_up, infer_p_strong_up]
```

Condition vector:

```text
technical features + regime one-hot + rationale forecast distribution + grounding scores
```

Flow objective:

```text
z0 ~ N(0, sigma_regime^2 I)
t ~ Uniform(0,1)
zt = t * z1 + (1 - t) * z0
loss = MSE(v_theta(t, zt, cond), z1 - z0)
```

Regime sigma:

```text
low_vol: 0.5
normal_vol: 1.0
high_vol: 1.5
```

## Tasks
Create:

```text
src/reward/flow_dataset.py
src/reward/flow_model_lite.py
src/reward/train_flow_reward_lite.py
src/reward/score_with_flow_reward.py
src/reward/evaluate_flow_reward.py
```

Architecture: MLP, hidden_dim 512, 4 layers, output_dim 5.

## Verification
Run:

```bash
cd firefin
python src/reward/train_flow_reward_lite.py \
  --judge-scores data/judge_outputs/judge_scores_h1.parquet \
  --tech-features data/indicators/technical_features_h1.parquet \
  --epochs 50 \
  --batch-size 256 \
  --output outputs/checkpoints/flow_reward_lite_h1.pt \
  --metrics outputs/metrics/flow_reward_lite_h1.json

python src/reward/score_with_flow_reward.py \
  --checkpoint outputs/checkpoints/flow_reward_lite_h1.pt \
  --judge-scores data/judge_outputs/judge_scores_h1.parquet \
  --tech-features data/indicators/technical_features_h1.parquet \
  --output data/judge_outputs/flow_rewards_h1.parquet
```

Then:

```bash
python - <<'PYCHECK'
import json, pandas as pd
m=json.load(open('outputs/metrics/flow_reward_lite_h1.json'))
print(m)
r=pd.read_parquet('data/judge_outputs/flow_rewards_h1.parquet')
print(r[['flow_prob_true_label','flow_entropy','flow_overall_reward']].describe())
assert r['flow_overall_reward'].notna().mean() > 0.95
PYCHECK
```

## Acceptance criteria
PASS only if checkpoint exists, flow rewards exist for at least 95% judged candidates, and validation MSE beats naive mean-target baseline.

## Status JSON

```json
{
  "step": "10_FLOW_REWARD_MODEL_LITE",
  "status": "PASS|FAIL",
  "checkpoint": "outputs/checkpoints/flow_reward_lite_h1.pt",
  "validation_mse": 0.0,
  "naive_baseline_mse": 0.0,
  "scored_rows": 0,
  "notes": "..."
}
```
