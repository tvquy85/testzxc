# Step 07.5 Story - Rationale Repetition Was Mostly Technical Vocabulary

The original Step 07 rationale audit blocked `rationale_quality` because overall within-sample similarity and repeated phrase rates were high. The top repeated phrases were technical feature names:

```text
macd bearish momentum
macd bullish momentum
price below 20-day sma
price above sma20
rsi overbought
```

Step 07.5 decomposed rationale wording into news/meta explanation text versus technical signal vocabulary.

Key result:

```text
news_plus_meta_mean_jaccard: 0.6447
news_plus_meta_template_cluster_rate: 0.2257
news_repeated_phrase_rate: 0.0000
technical_repeated_phrase_rate: 0.8230
evidence_citation_rate: 1.0000
news_rationale_empty_when_N_rate: 0.0082
```

Interpretation:

The free-form news/conflict/risk rationale wording is not template-heavy under the strict thresholds. Overall repetition is dominated by repeated technical indicator names, which are feature vocabulary rather than generic explanation templates.

Paper value:

- This gives a defensible way to report rationale quality without hiding technical-vocabulary repetition.
- It prevents a metric artifact from incorrectly blocking explanation quality.
- It still preserves scientific honesty by reporting the high technical repetition separately.

Potential paper wording:

```text
We decompose rationale repetition into explanatory wording and technical feature vocabulary. The news/conflict/risk rationale text satisfies diversity thresholds, while most repeated phrases correspond to deterministic technical indicator names. We therefore report technical-vocabulary repetition separately rather than treating it as generic rationale templating.
```
