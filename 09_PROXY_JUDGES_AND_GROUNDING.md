# Step 09 — Proxy Judges and Grounding Scores

> **Use with Antigravity / Gemini Pro 3.1 High**  
> Treat this file as one bounded task. Do not implement later steps unless this file explicitly asks for them.  
> Always create small scripts, run verification, and save a short status JSON before stopping.

## Goal
Score each candidate rationale using multiple cheap/local judges. Output becomes training data for the flow reward model.

## Inputs

```text
data/rationales/candidate_rationales_h1.jsonl
data/labels/aligned_samples_h1.parquet
data/indicators/technical_features_h1.parquet
data/indicators/technical_event_tokens_h1.parquet
prompts/proxy_inferability_judge_prompt.txt
prompts/financial_soundness_judge_prompt.txt
configs/local_paths.yaml
```

## Outputs

```text
data/judge_outputs/judge_scores_h1.parquet
outputs/status/09_PROXY_JUDGES_AND_GROUNDING.status.json
```

## Judges

- Decision inferability LLM: Qwen3-4B, DeepSeek-R1-Distill-Qwen-1.5B, or Qwen2.5-3B.
- Financial soundness LLM: FinGPT if loadable, otherwise Qwen3/Phi.
- Technical grounding rule-checker: deterministic Python.
- News NLI grounding: `cross-encoder--nli-deberta-v3-small`.
- Utility judge: long/short/hold payoff after transaction cost.

## Required output columns

```text
sample_id, candidate_id, label_5,
infer_p_strong_down, infer_p_mild_down, infer_p_neutral, infer_p_mild_up, infer_p_strong_up,
infer_pred_label, infer_prob_true_label,
financial_soundness_score, overconfidence_score,
technical_grounding_score, news_entailment_rate, news_contradiction_rate,
utility_score, overall_proxy_score
```

Overall proxy score MVP formula:

```text
0.40 * infer_prob_true_label
+ 0.20 * technical_grounding_score
+ 0.15 * news_entailment_rate
- 0.15 * news_contradiction_rate
+ 0.10 * normalized_utility_score
+ 0.10 * financial_soundness_score
```

Clip to `[0,1]`.

## Tasks
Create:

```text
src/judges/inferability_judge.py
src/judges/financial_soundness_judge.py
src/judges/technical_grounding_judge.py
src/judges/news_nli_grounding_judge.py
src/judges/utility_judge.py
src/judges/run_all_judges.py
```

## Verification
Run:

```bash
cd firefin
python src/judges/run_all_judges.py \
  --rationales data/rationales/candidate_rationales_h1.jsonl \
  --samples data/labels/aligned_samples_h1.parquet \
  --tech-features data/indicators/technical_features_h1.parquet \
  --tech-tokens data/indicators/technical_event_tokens_h1.parquet \
  --config configs/local_paths.yaml \
  --limit 3000 \
  --output data/judge_outputs/judge_scores_h1.parquet
```

Then:

```bash
python - <<'PYCHECK'
import pandas as pd
j=pd.read_parquet('data/judge_outputs/judge_scores_h1.parquet')
print(j.shape)
print(j[['infer_prob_true_label','technical_grounding_score','overall_proxy_score']].describe())
assert j['overall_proxy_score'].between(0,1).mean() > 0.95
assert j['sample_id'].nunique() >= 500
PYCHECK
```

## Acceptance criteria
PASS only if at least 500 unique samples are scored and `overall_proxy_score` is valid for at least 95% rows.

## Status JSON

```json
{
  "step": "09_PROXY_JUDGES_AND_GROUNDING",
  "status": "PASS|FAIL",
  "scored_rows": 0,
  "unique_samples": 0,
  "mean_overall_proxy_score": 0.0,
  "models_used": [],
  "notes": "..."
}
```
