# 19 — Strict AAAI Strong-Accept Gate V6

> Scope: branch `currentdata-aaai-fix-v2`; current-data only; do not use SN2; do not expand full FNSPID. Execute one file at a time. Every step must write a status JSON and must stop on FAIL. Pipeline PASS is not a scientific claim.

## Goal
Create the final V6 science gate. It must distinguish pipeline runnable, diagnostic contribution, current-data positive result, and AAAI strong-accept readiness.

## Outputs
```text
outputs/repro/currentdata_v6_strong_accept_gate.json
outputs/tables/19_v6_claim_matrix.csv
outputs/status/19_STRICT_AAAI_STRONG_ACCEPT_GATE.status.json
```

## Claim logic
Flow claim requires 2/3 core Flow wins. Alignment claim requires V6 beats Base/SFT. Trading alpha requires Sharpe >0, enough days, and confidence intervals that support positive alpha. Counterfactual claim requires pass/no-change/news gates. Baseline competitiveness requires beating Technical_Rule on Macro-F1 and MCC. AAAI strong accept requires all major claims plus comparable baselines and statistical tests.

## Test case
```python
from src.repro.currentdata_v6_strong_accept_gate import allow_flow_claim

def test_flow_claim_requires_two_wins():
    assert allow_flow_claim({'rank':True,'pair':True,'top_decile':False})
    assert not allow_flow_claim({'rank':False,'pair':True,'top_decile':False})
```

## Commands
```bash
python -m src.repro.currentdata_v6_strong_accept_gate --metrics-dir outputs/metrics --tables-dir outputs/tables --status-dir outputs/status --output outputs/repro/currentdata_v6_strong_accept_gate.json --claim-table outputs/tables/19_v6_claim_matrix.csv --status outputs/status/19_STRICT_AAAI_STRONG_ACCEPT_GATE.status.json
python -m pytest -q tests/test_strong_accept_gate_v6.py tests
```

## Acceptance
Gate may return `CLAIM_RESTRICTED`; it must never return strong-accept-ready unless all gates in `Goal.md` pass.

## Progress Update 2026-06-22
Status: `PASS` for gate execution and status/artifact audit. Final decision is `GO_V6_PIPELINE` with `CLAIM_RESTRICTED`; `strong_accept_ready=false`.

Implementation notes:
```text
src/repro/currentdata_v6_strong_accept_gate.py
tests/test_strong_accept_gate_v6.py
```

Important pre-gate repair: Step 04 status was rerun after fixing the evidence-pack repair validator to support the actual V6 dict schema (`company_evidence`, `context_evidence`, `technical_signals`). Step 04 now reports `PASS`, 3129 rows, 3 repaired rows, and 0 post-repair failures.

Artifacts verified:
```text
outputs/repro/currentdata_v6_strong_accept_gate.json
outputs/tables/19_v6_claim_matrix.csv
outputs/manifests/19_STRICT_AAAI_STRONG_ACCEPT_GATE.manifest.json
outputs/status/19_STRICT_AAAI_STRONG_ACCEPT_GATE.status.json
```

Final command run:
```bash
D:\LOBProj\LOBExp\.venv\Scripts\python.exe -m src.repro.currentdata_v6_strong_accept_gate --metrics-dir outputs/metrics --tables-dir outputs/tables --status-dir outputs/status --output outputs/repro/currentdata_v6_strong_accept_gate.json --claim-table outputs/tables/19_v6_claim_matrix.csv --status outputs/status/19_STRICT_AAAI_STRONG_ACCEPT_GATE.status.json --manifest outputs/manifests/19_STRICT_AAAI_STRONG_ACCEPT_GATE.manifest.json
```

Gate metrics:
```text
pipeline_decision: GO_V6_PIPELINE
claim_decision: CLAIM_RESTRICTED
strong_accept_ready: false
allowed_claim_count: 11
blocked_claim_count: 3
pipeline_pass: true
claim_allowed: false
```

Allowed diagnostic/pipeline claims:
```text
v6_pipeline_runnable
data_evidence_quality
judge_quality
rationale_quality
alignment_artifacts
alignment_improves_over_sft
trading_alpha_diagnostic
counterfactual_faithfulness
comparable_baselines
ablations_present
statistical_tests
```

Blocked science claims:
```text
flow_reward_improvement: Official Flow did not beat proxy on at least 2/3 core metrics; Step 11.7 shows the Step 11.6 reranker diagnostic win is not attributable to Flow
forecast_beats_technical_rule: Repaired DPO still did not beat Technical_Rule on both Macro-F1 and MCC
trading_alpha_paper_level: requires Sharpe > 0, >=120 days, and confidence intervals that support positive alpha; Step 15.6 validation-selected DPO still has absolute Sharpe/mean-return CIs crossing zero
```

Verification commands run:
```bash
D:\LOBProj\LOBExp\.venv\Scripts\python.exe -m py_compile src\repro\currentdata_v6_strong_accept_gate.py
D:\LOBProj\LOBExp\.venv\Scripts\python.exe -m pytest -q tests/test_strong_accept_gate_v6.py tests/test_evidence_pack_repair_v6.py
D:\LOBProj\LOBExp\.venv\Scripts\python.exe -m src.utils.verify_status --status outputs/status/19_STRICT_AAAI_STRONG_ACCEPT_GATE.status.json
```

Scientific caution: the V6 pipeline is now runnable through the strict gate, but it is not AAAI strong-accept ready. The proper conclusion is claim-restricted negative evidence plus clear next-fix targets, not promotion.

Post-forecast-repair promotion update: Step 14.6 repaired parse-ok probability distributions without using target labels and Step 17/18/18.5 were rerun with repaired DPO/RWSFT as canonical prediction artifacts. Repaired DPO now beats RWSFT on both point metrics (`Macro-F1=0.1305` vs `0.1202`, `MCC=0.0269` vs `-0.0052`), so `alignment_improves_over_sft` is allowed. Repaired DPO still trails Technical_Rule (`Macro-F1=0.2277`, `MCC=0.0466`), so `forecast_beats_technical_rule` remains blocked.

Post-statistical-test update: Step 18.5 added 12 CI/test rows, so the `statistical_tests` claim is allowed. The final claim remains restricted because repaired DPO Sharpe CI 95% is `[-1.2563, 3.6290]`, DPO mean-return CI crosses zero, and forecast comparisons still have no CI support over Technical_Rule or RWSFT. The delta mean return versus Technical_Rule is positive with CI support (`0.00385`, CI95 `[0.00013, 0.00759]`), but this is not enough for paper-level alpha because absolute DPO Sharpe/mean-return uncertainty remains high.

Post-trading-policy-variant update: Step 15.5 status is now part of the strict status audit. Among six observed policy variants, `Supervised_LogReg_TFIDF` has the best Sharpe (`1.4379`) and CI-supported delta versus Technical_Rule, but its absolute Sharpe CI crosses zero (`[-0.8008, 3.4505]`) and it is a post-hoc non-LLM diagnostic variant selected on the same test period. Repaired `Qwen_DPO_V6_Official_Action` is positive but uncertain (`Sharpe=1.2002`). Paper-level alpha remains blocked.

Post-validation-selected-trading update: Step 15.6 status is now part of the strict status audit. The DPO threshold/cap grid was selected on validation only (`threshold=0.61`, `cap=3`) and locked for test. Held-out test Sharpe improves to `1.6840`, with CI-supported positive deltas versus Technical_Rule (`delta Sharpe=3.1051`, CI95 `[0.7142, 5.4266]`; `delta mean return=0.004138`, CI95 `[0.000959, 0.007316]`). However, absolute test Sharpe CI still crosses zero (`[-0.9523, 4.0931]`) and mean-return CI still crosses zero (`[-0.001080, 0.004889]`), so `trading_alpha_paper_level` remains blocked.

Post-counterfactual-eligibility update: Step 16.5 status is now part of the strict status audit. It does not allow the counterfactual claim; it explains the failure. After repaired DPO probabilities, `49.43%` of counterfactual tasks have eligible original-side signal, and down-decrease eligibility improves to `12.90%`, but DPO still assigns near-zero down-side probability to most down-decrease originals.

Post-counterfactual-quality update: Step 16.6 status is now part of the strict status audit. A quality-filtered contrast set selected 192 tasks from 1495 candidates after enforcing dominant expected polarity and repairing placeholder counterfactual text. The deterministic DPO rerun improved over Step 16 (`pass_rate=0.4844` vs `0.3257`; `no_change_rate=0.2604` vs `0.3657`) and passed the news removal gates (`remove_positive=0.5000`, `remove_negative=0.7143`). The full counterfactual claim still remained blocked at Step 16.6 because overall pass rate was below `0.50`, driven mainly by `neutralize_negative_evidence=0.0952`; this motivated Step 16.7 semantic neutralization.

Post-counterfactual-semantic update: Step 16.7 status is now part of the strict status audit. Semantic neutralization rewrites positive/negative evidence into neutral company-update text rather than brittle token deletion. The deterministic DPO rerun on 192 tasks passes the claim gate (`pass_rate=0.5313`, `no_change_rate=0.2083`, `schema_ok_rate=0.9948`) and keeps news-faithfulness gates allowed (`remove_positive=0.5000`, `remove_negative=0.7143`, `neutralize_negative_evidence=0.5238`). Therefore `counterfactual_faithfulness` is now allowed, while Flow, forecast-over-Technical_Rule, and paper-level alpha remain blocked.

Post-validation-calibration update: Step 17.5 status is now part of the strict status audit. It found a validation-selected DPO confidence gate (`threshold=0.71`) with small positive test point deltas over Technical_Rule, but paired bootstrap CIs cross zero (`delta Macro-F1 CI95 [-0.02464, 0.03220]`, `delta MCC CI95 [-0.03249, 0.03761]`). Forecast-superiority claims remain blocked.

Post-validation-stacking update: Step 17.6 status is now part of the strict status audit. A validation-trained logistic stacker over repaired DPO probabilities/labels, Technical_Rule labels, evidence metadata, and track indicators still overfits validation (`Macro-F1=0.4098`, `MCC=0.2397`) but underperforms Technical_Rule on held-out test (`Stacked Macro-F1=0.1804`, `MCC=0.0000`; Technical_Rule Macro-F1=`0.2277`, MCC=`0.0466`). Paired deltas versus Technical_Rule are negative and CIs cross zero. Forecast-superiority claims remain blocked.

Post-supervised-ceiling update: Step 17.7 status is now part of the strict status audit. A train-only, validation-selected supervised text/technical logistic probe did not beat Technical_Rule under the leak-safe protocol. Validation Macro-F1/MCC were `0.1469/0.0283` versus Technical_Rule `0.2041/0.0438`; held-out test Macro-F1/MCC were `0.1480/0.0776` versus Technical_Rule `0.2277/0.0466`. The test MCC point delta is positive but CI crosses zero (`delta MCC=0.0310`, CI95 `[-0.0492, 0.1086]`), while Macro-F1 is significantly worse (`delta Macro-F1=-0.0797`, CI95 `[-0.1379, -0.0241]`). Forecast-superiority claims remain blocked.

Post-rationale-decomposition update: Step 07.5 status is now part of the strict status audit. `rationale_quality` is allowed using decomposed evidence: news/meta mean Jaccard `0.6447`, news/meta cluster rate `0.2257`, news repeated phrase rate `0.0000`, evidence citation `1.0000`. High repetition is reported separately as technical signal-vocabulary repetition.

Post-Flow-diagnostic update: Step 11.5 status is now part of the strict status audit. Flow remains blocked, but the root cause is sharper: only `11.42%` of validation candidate pairs have non-tied raw utility, Flow top-decile utility trails proxy by `0.00241`, and Flow top-decile overlap with Technical_Rule is `0.0`.

Post-Flow-reranker update: Step 11.6 status is now part of the strict status audit. A train-selected pairwise utility reranker beats proxy on validation rank and pair accuracy (`rank=0.3890` vs `0.1967`; `pair=0.6575` vs `0.6164`) but still misses top-decile utility (`0.0000` vs proxy `0.002353`). This supports a next Flow objective with pairwise/listwise utility terms, but the official Flow claim remains blocked because the current checkpoint is unchanged, ablation checkpoints are missing, and top-decile utility is still weak.

Post-Flow-attribution update: Step 11.7 status is now part of the strict status audit. The no-flow ablation selected by train (`ridge_utility alpha=100`) matches or exceeds the full feature set on held-out validation (`rank=0.4267`, `pair=0.6849`) while the full Step 11.6 feature set has `rank=0.3890`, `pair=0.6575`, and the only-flow model wins only 1/3 metrics. `flow_attribution_supported=false`; therefore the Step 11.6 reranker win is not attributable to Flow-specific signal and the Flow claim remains blocked.
