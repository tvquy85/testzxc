import json
import os

models = {
    "Qwen-1.5B (Original Base)": "h1",
    "Llama-3.2-3B (RWSFT+DPO)": "llama",
    "DeepSeek-R1-Distill-1.5B (Base)": "deepseek_base",
    "DeepSeek-R1-Distill-1.5B (RWSFT+DPO)": "deepseek"
}

report = []
report.append("# Comprehensive Statistical Report: Model Ablation Study\n")
report.append("This report compares the performance of the original Qwen 1.5B model against DeepSeek-R1-Distill-1.5B and Llama-3.2-3B across Prediction Accuracy, Financial Backtesting (Sharpe Ratio), and Counterfactual Consistency (CFR).\n")

report.append("## 1. Prediction Performance\n")
report.append("| Model | Accuracy | Macro F1 | MCC | Brier Score |")
report.append("|-------|----------|----------|-----|-------------|")
for model, suffix in models.items():
    path = f"outputs/metrics/final_prediction_metrics_{suffix}.json"
    if os.path.exists(path):
        with open(path) as f:
            d = json.load(f)
            report.append(f"| {model} | {d['accuracy']:.4f} | {d['macro_f1']:.4f} | {d['mcc']:.4f} | {d['brier_score_macro']:.4f} |")

report.append("\n## 2. Trading Backtest Performance (Hold & Long/Short)\n")
report.append("| Model | Cumulative Return | Sharpe Ratio | Max Drawdown |")
report.append("|-------|-------------------|--------------|--------------|")
for model, suffix in models.items():
    path = f"outputs/metrics/final_backtest_{suffix}.json"
    if os.path.exists(path):
        with open(path) as f:
            d = json.load(f)
            # Assuming format: {'cumulative_return': X, 'sharpe_ratio': Y, 'max_drawdown': Z}
            ret = d.get('cumulative_return', 0)
            sharpe = d.get('sharpe_ratio', 0)
            mdd = d.get('max_drawdown', 0)
            report.append(f"| {model} | {ret:.4f} | {sharpe:.4f} | {mdd:.4f} |")

report.append("\n## 3. Counterfactual Consistency (CFR)\n")
report.append("| Model | Remove News | Neutralize Tech | Flip Regime |")
report.append("|-------|-------------|-----------------|-------------|")
for model, suffix in models.items():
    path = f"outputs/metrics/final_cf_metrics_{suffix}.json"
    if os.path.exists(path):
        with open(path) as f:
            d = json.load(f)
            cfr_metrics = d.get("cfr_metrics", {})
            news = cfr_metrics.get("CF_NEWS_REMOVE", 0) * 100
            tech = cfr_metrics.get("CF_TECH_NEUTRALIZE", 0) * 100
            regime = cfr_metrics.get("CF_REGIME_FLIP", 0) * 100
            report.append(f"| {model} | {news:.2f}% | {tech:.2f}% | {regime:.2f}% |")

with open("outputs/metrics/Statistical_Report.md", "w") as f:
    f.write("\n".join(report))

print("Report compiled successfully.")
