# Step 18.5 Story - CI Blocks Alpha Claim Despite Positive Sharpe

Current-data V6 produced a positive point-estimate daily annualized Sharpe (`0.9667`) after costs, but moving-block bootstrap uncertainty was wide:

```text
DPO Sharpe CI 95%: [-1.5791, 3.3879]
DPO mean daily return CI 95%: [-0.00203, 0.00457]
Delta mean return vs Technical_Rule CI 95%: [-0.00023, 0.00745]
```

This is a useful paper story because it separates runnable trading diagnostics from a paper-level alpha claim. The model may have a promising point estimate, but the current out-of-sample horizon is too short/noisy to prove positive alpha. The strict gate therefore allows the `statistical_tests` evidence claim while continuing to block `trading_alpha_paper_level`.

Reusable framing:
- Report point estimates and confidence intervals together.
- Treat CI crossing zero as negative/insufficient evidence, not as a failed implementation.
- Use this as motivation for longer out-of-sample validation or a stronger prediction objective before any strong finance claim.

Potential paper wording:

```text
Although the V6 aligned model yielded a positive after-cost Sharpe on the current-data test calendar, block-bootstrap confidence intervals included zero. We therefore treat the backtest as diagnostic evidence only and keep paper-level alpha claims blocked under the strict claim gate.
```
