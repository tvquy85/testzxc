# Step 15.5 Story - Trading Policy Variant Probe V6

Date: 2026-06-22

## What We Tested
After the official DPO backtest showed positive diagnostic Sharpe but the statistical gate blocked paper-level alpha, we evaluated existing policy variants on the same 173-day current-data test calendar:

- Technical_Rule
- Qwen_DPO_V6_Official_Action
- Qwen_DPO_V6_Label
- Validation_Calibrated_Hybrid
- Validation_Stacked_Logistic
- Supervised_LogReg_TFIDF

The purpose was to understand trading utility sensitivity, not to select the best strategy for a paper claim.

Method caution is based on:

- Lo 2002, Sharpe-ratio inference: Sharpe estimates have sampling error.
- White 2000, Reality Check: repeated trading-rule search creates data-snooping risk.
- Bailey and Lopez de Prado, Deflated Sharpe Ratio: selection bias, non-normality, and backtest overfitting can inflate apparent performance.

## Result
The best Sharpe among the observed variants was the non-LLM supervised diagnostic policy:

```text
Supervised_LogReg_TFIDF Sharpe: 1.4379, CI95 [-0.8008, 3.4505]
Supervised_LogReg_TFIDF mean daily return: 0.001811, CI95 [-0.001012, 0.004623]
Supervised_LogReg_TFIDF delta Sharpe vs Technical: 2.8590, CI95 [0.1021, 5.4438]
Supervised_LogReg_TFIDF delta mean return vs Technical: 0.004136, CI95 [0.000386, 0.007864]
```

The official DPO policy remains positive but uncertain:

```text
Qwen_DPO_V6_Official_Action Sharpe: 0.9667, CI95 [-1.7345, 3.4628]
Qwen_DPO_V6_Official_Action delta Sharpe vs Technical: 2.3878, CI95 [-0.2876, 5.2403]
```

Technical_Rule is negative on this trading metric:

```text
Technical_Rule Sharpe: -1.4211, CI95 [-3.5464, 0.5454]
```

## Paper Value
This is a useful nuance: classification Macro-F1 and trading utility are not perfectly aligned. Sparse policies can have poor classification metrics yet better return profiles than an always-active technical rule.

However, this cannot support paper-level alpha because the variants are compared after observing the same held-out test period, and the best variant's absolute Sharpe/mean-return CIs still cross zero. This is exactly the scenario where data-snooping and deflated-Sharpe cautions matter.

## Reusable Lesson
For the next full-scale run, choose one trading policy using validation only, preregister it, and evaluate it on a fresh out-of-sample period. Do not promote a best-of-many test-period variant.
