# Step 14 — Final Reproducibility Package for AAAI-Style Submission

> **Use with Antigravity / Gemini Pro 3.1 High**  
> Treat this file as one bounded task. Do not implement later steps unless this file explicitly asks for them.  
> Always create small scripts, run verification, and save a short status JSON before stopping.

## Goal
Package code, configs, metrics, and a concise technical report so results can be reproduced and reviewed.

## Inputs

```text
outputs/status/*.status.json
outputs/tables/*.csv
outputs/figures/*
outputs/metrics/*.json
configs/*.yaml
src/**/*.py
prompts/*.txt
```

## Outputs

```text
outputs/repro/README_REPRO.md
outputs/repro/EXPERIMENT_MANIFEST.json
outputs/repro/MODEL_USAGE.md
outputs/repro/DATA_CARD_FIRE_FIN_LITE.md
outputs/repro/RESULTS_SUMMARY.md
outputs/repro/firefin_lite_repro.zip
outputs/status/14_FINAL_AAAI_REPRO_PACKAGE.status.json
```

## Tasks

Create:

- `README_REPRO.md`: hardware, environment, execution order, rerun commands.
- `EXPERIMENT_MANIFEST.json`: dataset, horizon, labels, models, artifacts, metrics.
- `MODEL_USAGE.md`: model roles.
- `DATA_CARD_FIRE_FIN_LITE.md`: selected tickers, date range, label distribution, leakage handling.
- `RESULTS_SUMMARY.md`: prediction, explanation, backtest, ablation summary.
- `firefin_lite_repro.zip`: configs, prompts, src, status, metrics, tables, figures, reproducibility docs.

Exclude raw data, large checkpoints, and HF cache.

## Verification
Run:

```bash
cd firefin
python - <<'PYCHECK'
from pathlib import Path
required = [
 'outputs/repro/README_REPRO.md',
 'outputs/repro/EXPERIMENT_MANIFEST.json',
 'outputs/repro/MODEL_USAGE.md',
 'outputs/repro/DATA_CARD_FIRE_FIN_LITE.md',
 'outputs/repro/RESULTS_SUMMARY.md',
 'outputs/repro/firefin_lite_repro.zip',
]
for p in required:
    print(p, Path(p).exists())
    assert Path(p).exists()
PYCHECK
```

## Acceptance criteria
PASS only if zip exists, excludes large artifacts, includes status files, and results summary states which claims are supported vs unsupported.

## Status JSON

```json
{
  "step": "14_FINAL_AAAI_REPRO_PACKAGE",
  "status": "PASS|FAIL",
  "repro_zip": "outputs/repro/firefin_lite_repro.zip",
  "included_files_count": 0,
  "excluded_large_artifacts": true,
  "notes": "..."
}
```
