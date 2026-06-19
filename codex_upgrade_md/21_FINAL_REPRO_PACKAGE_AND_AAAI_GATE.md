# 21 — Final reproducibility package and AAAI gate

## Goal

Prepare a reproducibility package and a strict go/no-go report for AAAI-level submission.

## Codex task

Create:

```text
src/repro/build_repro_package.py
src/repro/aaai_gate_check.py
```

## Repro package contents

```text
configs/
prompts/
src/
tests/
outputs/tables/final/
outputs/metrics/
outputs/manifests/
README_REPRODUCE.md
REPRODUCIBILITY_CHECKLIST_DRAFT.md
```

Do not include large raw datasets or model weights unless explicitly requested. Include scripts to recreate them.

## AAAI gate check

The gate must fail if:

```text
dummy tables exist
hard-coded local paths remain
test data appears in alignment datasets
backtest is per-news instead of daily portfolio
flow v2 evaluation missing
counterfactual directional eval missing
less than 3 core baselines run
status JSON missing or failed
```

## Outputs

```text
outputs/repro/firefin_repro_package.zip
outputs/repro/aaai_gate_report.json
outputs/repro/README_REPRODUCE.md
```

## Verification commands

```bash
python src/repro/aaai_gate_check.py --output outputs/repro/aaai_gate_report.json
python src/repro/build_repro_package.py --output outputs/repro/firefin_repro_package.zip
```

## Acceptance criteria

- Gate report contains `decision: GO|NO_GO`.
- If `NO_GO`, list exact blocking issues.
- Repro package can be unzipped and has README.
- README includes one-command smoke test and expected outputs.


## Status JSON contract

When this task is complete, write:

```json
{
  "step": "21_FINAL_REPRO_PACKAGE_AND_AAAI_GATE",
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
outputs/status/21_FINAL_REPRO_PACKAGE_AND_AAAI_GATE.status.json
```

`next_step_allowed` must be `false` if any acceptance criterion fails.
