# 18 — Strict AAAI Science Gate

> Scope: upgrade `tvquy85/testzxc` branch `upgrade-aaai-reproducibility` on the **current already-built dataset/artifacts first**. Do not expand to full FNSPID until the current-data gates pass.
> Codex rule: implement only this task, run verification, write/update status JSON, and do not silently PASS if required outputs are missing.


## Goal
Replace the previous `GO` gate with a stricter scientific gate. Pipeline completeness is not enough; claim permission must depend on evidence.

## Inputs
- `outputs/status/*.json`
- `outputs/metrics/current_v3_claim_matrix.json`
- metrics from previous steps
- `outputs/manifests/*.json`

## Output
- `outputs/repro/currentdata_science_gate_report_v2.json`
- `outputs/status/18_AAAI_SCIENCE_GATE_STRICT.status.json`

## Codex task
Create `src/repro/currentdata_science_gate_v2.py`.

## Gate structure
```json
{
  "pipeline_decision": "GO|NO_GO",
  "claim_decision": "CLAIM_ALLOWED|CLAIM_RESTRICTED",
  "allowed_claims": [],
  "blocked_claims": [],
  "blocking_issues": [],
  "warnings": [],
  "evidence_files": []
}
```

## Required logic
`pipeline_decision=GO` if all required outputs exist and statuses PASS.

`claim_decision=CLAIM_ALLOWED` only if at least one scientific claim is allowed. Otherwise `CLAIM_RESTRICTED`.

Blocked claims must include reason, e.g.:
```json
{"claim":"trading_alpha","reason":"Sharpe <= 0 or test_days < 60"}
```

## Run
```bash
python -m src.repro.currentdata_science_gate_v2 \
  --claim-matrix outputs/metrics/current_v3_claim_matrix.json \
  --output outputs/repro/currentdata_science_gate_report_v2.json \
  --status outputs/status/18_AAAI_SCIENCE_GATE_STRICT.status.json
```

## Verification
```bash
python - <<'PY'
import json
r=json.load(open("outputs/repro/currentdata_science_gate_report_v2.json"))
assert r["pipeline_decision"] in ["GO","NO_GO"]
assert r["claim_decision"] in ["CLAIM_ALLOWED","CLAIM_RESTRICTED"]
assert isinstance(r["blocked_claims"], list)
print(json.dumps(r, indent=2)[:2000])
PY
```

## Acceptance criteria
- Gate clearly separates reproducibility from scientific success.
- Negative results cannot be presented as positive claims.
- The report includes next recommended action if claim restricted.
