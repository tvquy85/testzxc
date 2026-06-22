# Goal — Current-Data FIRE-Fin V6 Recovery Toward AAAI 2027 Strong-Accept Criteria

> Scope: branch `currentdata-aaai-fix-v2`; current-data only; do not use SN2; do not expand full FNSPID. Execute one file at a time. Every step must write a status JSON and must stop on FAIL. Pipeline PASS is not a scientific claim.

## Why this goal exists
The current Clean V4 Medium run is reproducible but not AAAI-ready: the report shows `GO_MEDIUM` with `CLAIM_RESTRICTED`; Flow V5 does **not** beat proxy on enough metrics; backtest Sharpe is negative; counterfactual faithfulness is blocked; Technical_Rule is the best macro-F1 baseline; and PEN/SEP/Policy are only reference-only baselines. This package upgrades the current-data pipeline first, before any full-scale expansion.

## Strong-accept target, not current claim
A strong AAAI 2027 submission needs at least the following evidence on the current dataset before scale-up:

1. Data/evidence quality: hard-event subset is explicit, audited, and separated from general/listicle/context-only evidence.
2. Rationale quality: schema remains high, but rationales are not template-heavy and use company evidence when available.
3. Judge quality: independent judge exceeds random by a meaningful margin and remains stable under label-order permutations.
4. Flow reward: Flow beats proxy on at least 2/3 metrics on held-out validation and is evaluated against reward target and raw realized utility.
5. Alignment: RWSFT/DPO are trained on enough clean examples and improve over base/SFT.
6. Prediction: aligned model beats `Technical_Rule` on Macro-F1 and MCC; otherwise block forecast claim.
7. Backtest: any alpha claim requires positive after-cost Sharpe, >=120 trading days for paper-level evidence, and confidence intervals.
8. Counterfactual faithfulness: pass rate >=0.50, no-change <=0.35, and positive/negative news perturbations each >=0.35.
9. Baselines: current-data comparable baselines include Technical_Rule, text baseline, Qwen base/SFT/RWSFT/DPO, SEP-style, and Policy-style proxy reward.
10. Claim gating: unsupported claims remain blocked.

## Required final artifacts
```text
outputs/status/*.status.json
outputs/metrics/*.json
outputs/tables/*.csv
outputs/repro/currentdata_v6_strong_accept_gate.json
review_samples/currentdata_v6/*.jsonl|*.csv|*.md
```

## Universal test command
```bash
python -m pytest -q tests
python -m src.utils.verify_status outputs/status/<STEP>.status.json
```

## Universal status contract
```json
{
  "step": "STEP_NAME",
  "status": "PASS|FAIL",
  "inputs": ["..."],
  "outputs": ["..."],
  "metrics": {},
  "failures": [],
  "next_step_allowed": true
}
```

## Stop rules
Stop immediately if any required input artifact is missing; schema/parse rate is below gate; tests fail; status JSON contract fails; or a claim is marked allowed without evidence.
