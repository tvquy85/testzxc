# FIRE-Fin Reproducibility Package

This package contains scripts, configs, prompts, tests, metrics, manifests, and final table artifacts for the FIRE-Fin AAAI upgrade audit.

Current gate decision: `GO`.

Blocking issues:

- none

One-command smoke test from repo root:

```bash
/mnt/d/LOBProj/LOBExp/.venv/Scripts/python.exe -m pytest -q tests
```

Expected smoke output for this snapshot:

```text
25 passed
```

Gate check:

```bash
/mnt/d/LOBProj/LOBExp/.venv/Scripts/python.exe src/repro/aaai_gate_check.py --output outputs/repro/aaai_gate_report.json
```

Expected gate output is `GO` only when all status files pass and the final blocker checks pass.

Large raw datasets and model weights are intentionally excluded. Use the scripts and local `$HF_HOME` configuration to recreate data/model-dependent artifacts.
