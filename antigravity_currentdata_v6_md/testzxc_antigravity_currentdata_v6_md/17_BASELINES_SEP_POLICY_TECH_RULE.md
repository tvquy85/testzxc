# 17 — Comparable Baselines: Technical Rule, SEP-Style, Policy-Style Proxy

> Scope: branch `currentdata-aaai-fix-v2`; current-data only; do not use SN2; do not expand full FNSPID. Execute one file at a time. Every step must write a status JSON and must stop on FAIL. Pipeline PASS is not a scientific claim.

## Goal
Stop relying on reference-only PEN/SEP/Policy labels. Add comparable current-data baselines where feasible.

## Required baselines
Technical_Rule, Qwen_Base_NoAlign, Qwen_RWSFT_V6, Qwen_DPO_V6, SEP_Style_Summarize_Explain, Policy_Style_Scalar_Proxy, PEN_Reference_Only if no current-data PEN adapter is implemented.

## Outputs
```text
outputs/tables/17_v6_comparable_baselines.csv
outputs/metrics/17_v6_baseline_comparison.json
outputs/status/17_BASELINES_SEP_POLICY_TECH_RULE.status.json
```

## Test case
```python
import pandas as pd
from pathlib import Path

def test_baseline_table_has_required_methods():
    p=Path('outputs/tables/17_v6_comparable_baselines.csv')
    assert p.exists()
    methods=set(pd.read_csv(p)['method'])
    for name in ['Technical_Rule','Qwen_DPO_V6','Policy_Style_Scalar_Proxy']:
        assert name in methods
```

## Commands
```bash
python -m src.baselines.run_v6_comparable_baselines --contexts data/processed/ticker_date_evidence_contexts_h1_v6_repaired.parquet --dpo-predictions outputs/predictions/current_v6_dpo_predictions.parquet --rwsft-predictions outputs/predictions/current_v6_rwsft_predictions.parquet --output outputs/tables/17_v6_comparable_baselines.csv --metrics outputs/metrics/17_v6_baseline_comparison.json --status outputs/status/17_BASELINES_SEP_POLICY_TECH_RULE.status.json
python -m pytest -q tests/test_baseline_table_v6.py tests
```

## Acceptance
At least 6 baselines; reference-only baselines explicitly marked; forecast claim blocked unless V6 beats Technical_Rule on Macro-F1 and MCC.

## Progress Update 2026-06-22
Status: `PASS` for comparable-baseline table/artifact validity; status file reports `next_step_allowed=true`. Forecast-improvement claim remains blocked.

Implementation notes:
```text
src/baselines/run_v6_comparable_baselines.py
tests/test_baseline_table_v6.py
```

The Step 17 runner evaluates all comparable rows on the same 300 Step 14 prediction contexts. Rows without a current-data artifact are explicitly marked `reference_only=True` and excluded from outperform claims.

Artifacts verified:
```text
outputs/tables/17_v6_comparable_baselines.csv
outputs/metrics/17_v6_baseline_comparison.json
outputs/manifests/17_BASELINES_SEP_POLICY_TECH_RULE.manifest.json
outputs/status/17_BASELINES_SEP_POLICY_TECH_RULE.status.json
```

Final command run:
```bash
D:\LOBProj\LOBExp\.venv\Scripts\python.exe -m src.baselines.run_v6_comparable_baselines --contexts data/processed/current_v6_prediction_contexts.parquet --dpo-predictions outputs/predictions/current_v6_dpo_predictions.parquet --rwsft-predictions outputs/predictions/current_v6_rwsft_predictions.parquet --output outputs/tables/17_v6_comparable_baselines.csv --metrics outputs/metrics/17_v6_baseline_comparison.json --status outputs/status/17_BASELINES_SEP_POLICY_TECH_RULE.status.json --manifest outputs/manifests/17_BASELINES_SEP_POLICY_TECH_RULE.manifest.json
```

Baseline table summary:
```text
Technical_Rule: rows=300, Macro-F1=0.2277, MCC=0.0466
Qwen_DPO_V6: rows=300, Macro-F1=0.1305, MCC=0.0269, schema_ok_rate=1.0000
Qwen_RWSFT_V6: rows=300, Macro-F1=0.1202, MCC=-0.0052, schema_ok_rate=1.0000
Qwen_Base_NoAlign: reference_only=True, no current-data base prediction artifact
SEP_Style_Summarize_Explain: rows=300, Macro-F1=0.1672, MCC=0.0000
Policy_Style_Scalar_Proxy: rows=300, Macro-F1=0.1936, MCC=0.0357
Neutral_Hold_Majority: rows=300, Macro-F1=0.0667, MCC=0.0000
PEN_Reference_Only: reference_only=True, no current-data PEN adapter/prediction artifact
```

Final metrics:
```text
baseline_count: 8
comparable_baseline_count: 6
reference_only_count: 2
required_methods_present: true
best_method_by_macro_f1: Technical_Rule
best_macro_f1: 0.2277
technical_rule_macro_f1: 0.2277
technical_rule_mcc: 0.0466
qwen_dpo_macro_f1: 0.1305
qwen_dpo_mcc: 0.0269
dpo_beats_technical_rule_macro_f1: false
dpo_beats_technical_rule_mcc: false
forecast_claim_allowed: false
```

Acceptance result:
```text
status JSON gate: PASS
baseline count gate: PASS (8 >= 6)
required methods gate: PASS
reference-only marking gate: PASS
forecast claim gate: FAIL/BLOCKED (DPO does not beat Technical_Rule on Macro-F1 or MCC)
```

Verification commands run:
```bash
D:\LOBProj\LOBExp\.venv\Scripts\python.exe -m py_compile src\baselines\run_v6_comparable_baselines.py
D:\LOBProj\LOBExp\.venv\Scripts\python.exe -m src.utils.verify_status --status outputs/status/17_BASELINES_SEP_POLICY_TECH_RULE.status.json
D:\LOBProj\LOBExp\.venv\Scripts\python.exe -m pytest -q tests/test_baseline_table_v6.py
```

Post-Step-14.6 update: Step 17 now uses the repaired DPO/RWSFT prediction artifacts as the canonical prediction inputs. Repaired DPO beats repaired RWSFT on both Macro-F1 and MCC, opening the alignment-over-RWSFT point claim in Step 19. Technical_Rule remains the strongest Macro-F1 and MCC baseline, so the forecast-superiority claim stays blocked.

Scientific caution: Step 17 is a valid negative forecast baseline comparison. V6 DPO/RWSFT artifacts are comparable on current-data rows, but Technical_Rule remains the strongest Macro-F1 baseline. Any forecast-superiority claim must stay blocked unless a later artifact beats Technical_Rule on both Macro-F1 and MCC with statistical support.
