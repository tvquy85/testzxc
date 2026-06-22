# Step 12 Story - Proxy-Gated Alignment Data Instead of Flow-Reward Overclaim

Date: 2026-06-22

Step 12 converted the negative Step 11 Flow result into a cleaner alignment-data decision. Since Flow V6 did not beat proxy strongly enough and `flow_claim_allowed=false`, alignment pair construction used the independent judge proxy rather than forcing a learned reward score that had not earned the claim.

The useful paper story is the gate discipline: a failed Flow reward did not stop the pipeline, but it changed the reward source. RWSFT/DPO examples were built from strict-valid rationale and judge rows only: rationale parse/schema OK, judge schema OK under all label-order variants, finite probability simplex, and train split only.

The first diagnostic was close but insufficient. Raw true-label probability plus naive JSON-token Jaccard produced fewer than 300 DPO pairs at the required `reward_gap >= 0.05` and `semantic_distance >= 0.15`. The fix was not to lower thresholds. Instead, semantic distance was computed over content tokens while excluding JSON schema/control tokens such as `news_rationale`, `forecast_distribution`, `action`, and label names. This better matches the intent of "semantic diversity" because boilerplate schema tokens should not make two completions look artificially similar.

Final Step 12 gate:

```text
rwsft_examples: 2976
dpo_pairs: 316
dpo_unique_samples: 206
reward_source: proxy_true_label_probability_ensemble
flow_scores_used: false
mean_reward_gap: 0.1737
min_observed_reward_gap: 0.0500
mean_semantic_distance: 0.3560
min_observed_semantic_distance: 0.1538
train_only: true
status: PASS
```

Paper angle to preserve: the alignment data is not presented as proof that the aligned model improves forecasts. It is a reproducible bridge from an audited judge signal to preference training. The next claim gate remains empirical: train RWSFT/DPO adapters, then test whether aligned models beat base/SFT and `Technical_Rule` on Macro-F1/MCC before allowing any forecast claim.

Potential wording:

```text
When the learned Flow reward failed its held-out utility gate, we did not use it as an alignment target. Instead, we constructed preference pairs from the independently calibrated judge distribution and required both a minimum reward gap and content-level semantic distance. This produces a conservative preference dataset: it is large enough for RWSFT/DPO, but still blocks downstream claims until adapter training and prediction gates pass.
```
