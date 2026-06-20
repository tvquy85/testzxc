# 00 — Master Execution Order: Current-Data First Upgrade

> Scope: upgrade `tvquy85/testzxc` branch `upgrade-aaai-reproducibility` on the **current already-built dataset/artifacts first**. Do not expand to full FNSPID until the current-data gates pass.
> Codex rule: implement only this task, run verification, write/update status JSON, and do not silently PASS if required outputs are missing.


## Goal
Make the current evaluation scientifically valid and then improve the weakest stages. Do not scale data yet.

## Known current bottlenecks
- 156,098 aligned rows, but news quality/entity matching is weak.
- Rationale parse/schema is high, but grounding is weak.
- Inferability is too close to self-declared forecast distribution.
- Flow v2 does not beat proxy.
- Counterfactual pass rate is low and no-change rate high.
- Daily backtest is negative and short.
- Ablations are NOT_RUN.

## Strict rules
1. Create a new branch from `upgrade-aaai-reproducibility`.
2. Write all new outputs with suffix `_v3` or `current_v3`.
3. Do not overwrite old artifacts.
4. Every task writes `outputs/status/<STEP>.status.json`.
5. PASS only if required outputs exist and row counts are nonzero.
6. Scientific gates must separate `pipeline_pass` from `claim_allowed`.
7. Train/reward/alignment use train only.
8. Test is for final evaluation only.

## Final artifact
`outputs/repro/currentdata_science_gate_report_v2.json`:
```json
{
  "pipeline_decision": "GO|NO_GO",
  "claim_decision": "CLAIM_ALLOWED|CLAIM_RESTRICTED",
  "allowed_claims": [],
  "blocked_claims": [],
  "blocking_issues": []
}
```
