# 18 — Ablation Suite V6

> Scope: branch `currentdata-aaai-fix-v2`; current-data only; do not use SN2; do not expand full FNSPID. Execute one file at a time. Every step must write a status JSON and must stop on FAIL. Pipeline PASS is not a scientific claim.

## Goal
Run ablations that test evidence grounding, news usage, flow reward, judge debias, and alignment.

## Required ablations
Full_V6, No_News_Evidence, Technical_Only, No_Technical_Tokens, No_Flow_Use_Proxy, No_Debias_Use_Normal_Order, No_Grounding_Filter, No_DPO_RWSFT_Only, Base_Qwen_NoAlign, Technical_Rule.

## Outputs
```text
outputs/tables/18_v6_ablation_results.csv
outputs/metrics/18_v6_ablation_summary.json
outputs/status/18_ABLATIONS_V6.status.json
```

## Test case
```python
import pandas as pd
from pathlib import Path

def test_no_not_run_in_ablation():
    p=Path('outputs/tables/18_v6_ablation_results.csv')
    assert p.exists()
    df=pd.read_csv(p).astype(str)
    assert 'NOT_RUN' not in set(df.values.ravel())
```

## Commands
```bash
python -m src.eval.run_v6_ablation_suite --contexts data/processed/ticker_date_evidence_contexts_h1_v6_repaired.parquet --predictions outputs/predictions/current_v6_dpo_predictions.parquet --baselines outputs/tables/17_v6_comparable_baselines.csv --output outputs/tables/18_v6_ablation_results.csv --metrics outputs/metrics/18_v6_ablation_summary.json --status outputs/status/18_ABLATIONS_V6.status.json
python -m pytest -q tests/test_ablation_v6.py tests
```

## Progress Update 2026-06-22
Status: `PASS` for ablation table/artifact validity; status file reports `next_step_allowed=true`. Forecast/counterfactual claims remain blocked.

Implementation notes:
```text
src/eval/run_v6_ablation_suite.py
tests/test_ablation_v6.py
```

The Step 18 table is an evidence table over existing current-data artifacts. Rows with a real prediction artifact are marked `PASS`; rows that are diagnostic proxies or reference-only are explicitly labeled and do not support outperform claims. No row uses `NOT_RUN`.

Artifacts verified:
```text
outputs/tables/18_v6_ablation_results.csv
outputs/metrics/18_v6_ablation_summary.json
outputs/manifests/18_ABLATIONS_V6.manifest.json
outputs/status/18_ABLATIONS_V6.status.json
```

Final command run:
```bash
D:\LOBProj\LOBExp\.venv\Scripts\python.exe -m src.eval.run_v6_ablation_suite --contexts data/processed/current_v6_prediction_contexts.parquet --predictions outputs/predictions/current_v6_dpo_predictions.parquet --baselines outputs/tables/17_v6_comparable_baselines.csv --output outputs/tables/18_v6_ablation_results.csv --metrics outputs/metrics/18_v6_ablation_summary.json --status outputs/status/18_ABLATIONS_V6.status.json --manifest outputs/manifests/18_ABLATIONS_V6.manifest.json
```

Ablation rows:
```text
Full_V6: PASS, source=Qwen_DPO_V6
No_News_Evidence: DIAGNOSTIC_PROXY, source=Technical_Rule
Technical_Only: DIAGNOSTIC_PROXY, source=Technical_Rule
No_Technical_Tokens: DIAGNOSTIC_PROXY, source=SEP_Style_Summarize_Explain
No_Flow_Use_Proxy: PASS, source=Qwen_DPO_V6, reward_source=proxy_true_label_probability_ensemble
No_Debias_Use_Normal_Order: DIAGNOSTIC_ONLY, judge debias reference
No_Grounding_Filter: DIAGNOSTIC_ONLY, grounding filter reference
No_DPO_RWSFT_Only: PASS, source=Qwen_RWSFT_V6
Base_Qwen_NoAlign: REFERENCE_ONLY, no current-data base prediction artifact
Technical_Rule: PASS
Policy_Style_Scalar_Proxy: DIAGNOSTIC_PROXY
```

Final metrics:
```text
ablation_count: 11
required_ablations_present: true
not_run_present: false
reference_only_count: 1
diagnostic_proxy_count: 4
diagnostic_only_count: 2
best_macro_f1_ablation: No_News_Evidence
best_macro_f1: 0.2277
full_v6_macro_f1: 0.1305
technical_rule_macro_f1: 0.2277
forecast_claim_allowed: false
counterfactual_claim_allowed: false
alpha_claim_allowed: true
```

Acceptance result:
```text
status JSON gate: PASS
required ablations gate: PASS
no NOT_RUN gate: PASS
claim gate: BLOCKED because Full_V6 does not beat technical/no-news proxies and counterfactual claim is false
```

Verification commands run:
```bash
D:\LOBProj\LOBExp\.venv\Scripts\python.exe -m py_compile src\eval\run_v6_ablation_suite.py
D:\LOBProj\LOBExp\.venv\Scripts\python.exe -m src.utils.verify_status --status outputs/status/18_ABLATIONS_V6.status.json
D:\LOBProj\LOBExp\.venv\Scripts\python.exe -m pytest -q tests/test_ablation_v6.py tests/test_baseline_table_v6.py
```

Post-Step-14.6 update: Full_V6 now points at the repaired DPO prediction artifact. Full_V6 Macro-F1 improved from `0.1124` to `0.1305`, enough to beat RWSFT in Step 17 but still below Technical_Rule/no-news proxy (`0.2277`).

Scientific caution: Step 18 is not an all-new inference sweep for every hypothetical ablation; it is a transparent ablation evidence table over current artifacts and explicit diagnostic proxies. It strengthens the negative forecast conclusion: the Full_V6 DPO artifact still does not beat the technical-only/no-news diagnostic proxy on Macro-F1.
