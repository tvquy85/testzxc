# Step 15 - Calendar-complete daily backtest

Step 15 exposed a useful methodological guardrail for the paper: a backtest can
look materially stronger if Sharpe is computed only on active trading days. The
first V6 daily-backtest run returned 70 active days and Sharpe 1.5174, but this
was not acceptable because the Step 14 prediction-context artifact spans 173
test calendar trading days.

The runner was corrected before accepting Step 15. Non-position days are now
included as zero-return portfolio days. The accepted Step 15 artifact reports
173 trading days, 70 nonzero-position days, Sharpe 0.9667 after costs, and
Technical_Rule Sharpe -1.4211 on the same calendar.

Paper value:
- The method section should state that daily portfolio returns are computed on
  the full evaluation calendar, not only on days with nonzero positions.
- The result is useful as an honest after-cost diagnostic, but not yet an alpha
  claim. Step 14 classification metrics remain weak, and Step 17/18/19 baseline,
  ablation, and statistical gates still control paper claims.
- The selected 300-row prediction context has no no-news subset, so any no-news
  ablation claim must wait for a dedicated Step 18 artifact.
