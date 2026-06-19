# Comprehensive Statistical Report: Model Ablation Study

This report compares the performance of the original Qwen 1.5B model against DeepSeek-R1-Distill-1.5B and Llama-3.2-3B across Prediction Accuracy, Financial Backtesting (Sharpe Ratio), and Counterfactual Consistency (CFR).

## 1. Prediction Performance

| Model | Accuracy | Macro F1 | MCC | Brier Score |
|-------|----------|----------|-----|-------------|
| Qwen-1.5B | 0.3100 | 0.1411 | 0.0715 | 0.2551 |
| DeepSeek-R1-Distill-1.5B | 0.1300 | 0.1170 | 0.0534 | 0.3142 |
| Llama-3.2-3B | 0.2400 | 0.1427 | 0.0558 | 0.2909 |

## 2. Trading Backtest Performance (Hold & Long/Short)

| Model | Cumulative Return | Sharpe Ratio | Max Drawdown |
|-------|-------------------|--------------|--------------|
| Qwen-1.5B | 0.0000 | 4.0136 | -0.2268 |
| DeepSeek-R1-Distill-1.5B | 0.0000 | 4.0130 | -0.1707 |
| Llama-3.2-3B | 0.0000 | 4.9072 | -0.1894 |

## 3. Counterfactual Consistency (CFR)

| Model | Remove News | Neutralize Tech | Flip Regime |
|-------|-------------|-----------------|-------------|
| Qwen-1.5B | 48.00% | 50.00% | 48.00% |
| DeepSeek-R1-Distill-1.5B | 56.00% | 68.00% | 50.00% |
| Llama-3.2-3B | 54.00% | 40.00% | 36.00% |