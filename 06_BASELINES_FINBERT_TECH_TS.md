# Step 06 — Baselines: FinBERT, Technical Features, and Time-Series Models

> **Use with Antigravity / Gemini Pro 3.1 High**  
> Treat this file as one bounded task. Do not implement later steps unless this file explicitly asks for them.  
> Always create small scripts, run verification, and save a short status JSON before stopping.

## Goal
Build strong lightweight baselines before training LLM rationales. This verifies whether the data has predictive signal.

## Inputs

```text
data/labels/aligned_samples_h1.parquet
data/indicators/technical_features_h1.parquet
data/indicators/technical_event_tokens_h1.parquet
configs/local_paths.yaml
```

## Outputs

```text
outputs/metrics/baseline_metrics_h1.json
outputs/tables/baseline_metrics_h1.csv
outputs/status/06_BASELINES_FINBERT_TECH_TS.status.json
```

## Tasks

### 1. Create scripts

```text
src/baselines/run_finbert_baseline.py
src/baselines/run_technical_lightgbm.py
src/baselines/run_combined_baseline.py
src/eval/classification_metrics.py
src/eval/calibration_metrics.py
```

### 2. Time split
Use time split only:

```text
train: earliest 70% dates
validation: next 15% dates
test: latest 15% dates
```

Save split assignment:

```text
data/processed/split_h1.parquet
```

### 3. FinBERT baseline
Use local `ProsusAI--finbert`.

Input: `headline + body` truncated to model max length.

Output features:

```text
finbert_positive, finbert_negative, finbert_neutral
```

Train classifier:

```text
LogisticRegression or LightGBM
```

### 4. Technical baseline
Use numeric technical features with LightGBM.

### 5. Combined baseline
Concatenate:

```text
FinBERT probs + technical features + regime one-hot
```

Train LightGBM.

### 6. Optional time-series baseline
If feasible within this step, run one of:

- `MOMENT-1-small` frozen embedding + MLP.
- `chronos-bolt-small` zero-shot price-only direction.
- `granite-timeseries-ttm-r2` zero/few-shot baseline.

If not feasible, create placeholder script and mark as `deferred` in status.

## Metrics
Compute:

```text
accuracy
macro_f1
mcc
brier_score_multiclass
expected_calibration_error
class_distribution
```

## Verification
Run:

```bash
cd firefin
python src/baselines/run_finbert_baseline.py --limit 20000 --output outputs/metrics/finbert_baseline_h1.json
python src/baselines/run_technical_lightgbm.py --limit 20000 --output outputs/metrics/technical_lgbm_h1.json
python src/baselines/run_combined_baseline.py --limit 20000 --output outputs/metrics/combined_lgbm_h1.json
```

Then:

```bash
python - <<'PYCHECK'
import json, glob
for p in glob.glob('outputs/metrics/*baseline*h1.json') + glob.glob('outputs/metrics/*lgbm*h1.json'):
    d=json.load(open(p))
    print(p, d.get('test_macro_f1'), d.get('test_mcc'))
PYCHECK
```

## Acceptance criteria
PASS only if:

- At least 3 baseline metrics JSON files exist.
- Train/val/test split file exists.
- Macro-F1 and MCC are reported.
- Combined baseline does not crash and is compared against majority-class baseline.

## Status JSON

```json
{
  "step": "06_BASELINES_FINBERT_TECH_TS",
  "status": "PASS|FAIL",
  "baseline_files": [],
  "best_baseline_by_macro_f1": "...",
  "best_macro_f1": 0.0,
  "notes": "..."
}
```
