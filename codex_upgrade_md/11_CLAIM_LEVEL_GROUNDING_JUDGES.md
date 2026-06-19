# 11 — Claim-level news and technical grounding judges

## Goal

Replace token-mention grounding with claim-level grounding.

## Existing problems

- Technical grounding currently checks whether token names are mentioned.
- News NLI should not use only headline as premise when rationale claims facts from technical data or market context.

## Files to create

```text
src/judges/extract_rationale_claims.py
src/judges/claim_level_grounding.py
src/judges/run_claim_grounding.py
```

## Codex task

1. Extract claims from parsed rationale into categories:

```text
news_claim
technical_claim
regime_claim
forecast_claim
risk_claim
```

2. For each technical claim, verify against structured technical token JSON and raw indicator values.
3. For each news claim, verify against headline + article body if available.
4. Use NLI cross-encoder only for news-text entailment/contradiction.
5. Do not give score 1.0 when no technical tokens exist. Use:

```text
not_applicable
```

## Outputs

```text
data/judges/claim_grounding_scores.parquet
outputs/metrics/claim_grounding_summary.json
```

## Verification commands

```bash
python src/judges/run_claim_grounding.py \
  --rationales data/rationales/parsed/train_candidates_strict.parquet \
  --tokens data/indicators/technical_event_tokens_h1_v2.parquet \
  --output data/judges/claim_grounding_scores.parquet \
  --summary outputs/metrics/claim_grounding_summary.json
```

## Acceptance criteria

- Grounding output has one row per claim, not only one row per rationale.
- Technical contradiction examples are saved.
- News contradiction examples are saved.
- Not-applicable is separate from perfect score.


## Status JSON contract

When this task is complete, write:

```json
{
  "step": "11_CLAIM_LEVEL_GROUNDING_JUDGES",
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
outputs/status/11_CLAIM_LEVEL_GROUNDING_JUDGES.status.json
```

`next_step_allowed` must be `false` if any acceptance criterion fails.
