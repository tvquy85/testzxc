# Sources and Review Basis

## Repository sources used
- GitHub branch: `tvquy85/testzxc`, branch `currentdata-aaai-fix-v2`.
- Current medium report: `BaoCaoclean_v4_medium_21062026.md`.
- Master medium order: `clean_v4_medium_recovery_codex_md/00_MASTER_CLEAN_V4_MEDIUM_ORDER.md`.
- Science gate: `review_samples/clean_v4_medium_21062026/24_science_gate_report.json`.
- Code inspected: `src/repro/currentdata_clean_v4_medium_science_gate.py`, `src/reward/build_flow_dataset_v5.py`, `src/judges/independent_inferability_judge_v4.py`, `src/eval/backtest_daily_portfolio_v3.py`.

## External references
- AAAI-27 Main Technical Track Call: official criteria/reproducibility context. URL: https://aaai.org/conference/aaai/aaai-27/main-technical-track-call/
- FNSPID: 29.7M stock records, 15.7M aligned news records, 4,775 companies, 1999–2023. URL: https://github.com/Zdong104/FNSPID_Financial_News_Dataset
- SEP: Learning to Generate Explainable Stock Predictions using Self-Reflective Large Language Models. URL: https://arxiv.org/abs/2402.03659
- BenchStock: realistic stock forecasting evaluation and portfolio metrics. URL: https://openreview.net/pdf/de87306fbdbad74acf00a17ebc34a3e3688998e3.pdf
- Policy paper in local repo/upload: `Translate Policy to Language: Flow Matching Generated Rewards for LLM Explanations`; key ideas include hidden decision, proxy LLM logits, rectified flow reward, sentence-level dense reward, and PPO/LoRA.
- Flow Reward V6 novelty note: `FLOW_REWARD_V6_NOVELTY_PROPOSAL.md`; key ideas include debiased judge distribution targets, reliability weighting, decision-utility validation, evidence-faithfulness metrics, and negative-result-aware claim gates.

## Additional references for Flow Reward V6 novelty
- Direct Preference Optimization: stable preference optimization without explicit PPO loop. URL: https://arxiv.org/abs/2305.18290
- RewardBench: reward-model evaluation should use structured prompt/chosen/rejected comparisons and expose RM limitations. URL: https://arxiv.org/abs/2403.13787
- Decision-Focused Learning: train/evaluate ML systems by decision quality under uncertainty, not standalone predictive fit. URL: https://arxiv.org/abs/2307.13565
- Faithful Model Explanation in NLP: explanation faithfulness must be separated from plausibility and can be assessed through counterfactual interventions. URL: https://arxiv.org/html/2209.11326v4
- LLM-as-a-judge position bias: label/order permutations are necessary because judge preferences can change under position swaps. URL: https://arxiv.org/html/2406.07791v7
- The Statistics of Sharpe Ratios: Sharpe estimates require statistical care and should not be over-claimed from short/noisy samples. URL: https://rpc.cfainstitute.org/research/financial-analysts-journal/2002/the-statistics-of-sharpe-ratios
- Stacked Generalization: ensemble meta-learning can combine model outputs, but held-out validation/test discipline is required before claiming generalization. DOI: https://doi.org/10.1016/S0893-6080(05)80023-1
- Scikit-learn StackingClassifier: final estimator uses base estimator outputs as inputs; useful as method reference for Step 17.6. URL: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.StackingClassifier.html
- Scikit-learn LogisticRegression: regularized logistic regression is the Step 17.6 meta-classifier. URL: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
- Scikit-learn model selection and cross-validation: validation-selected hyperparameters must be evaluated on held-out data. URL: https://scikit-learn.org/stable/modules/cross_validation.html
- Scikit-learn TimeSeriesSplit: time-ordered data should not train on future observations and evaluate on past observations. URL: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html
- Lopez de Prado financial ML validation lectures: purging/embargoing are used to reduce leakage in financial labels that depend on future horizons. URL: https://ssrn.com/abstract=3257420
- White 2000 Reality Check: repeated search over trading rules creates data-snooping risk. URL: https://www.ssc.wisc.edu/~bhansen/718/White2000.pdf
- Bailey and Lopez de Prado Deflated Sharpe Ratio: Sharpe should be corrected for selection bias, non-normality, and backtest overfitting. URL: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2460551
- Politis and Romano 1994 Stationary Bootstrap: dependent time-series resampling should preserve local dependence when estimating uncertainty. URL: https://www.ssc.wisc.edu/~bhansen/718/Politis%20Romano.pdf
- Bradley and Terry 1952 paired comparisons: pairwise preference data can estimate ranking strength. URL: https://academic.oup.com/biomet/article-abstract/39/3-4/324/326091
- Burges 2010 RankNet/LambdaRank/LambdaMART: learning-to-rank objectives target ordering and top-of-list quality. URL: https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/MSR-TR-2010-82.pdf
- Scikit-learn Ridge: L2-regularized regression is used as a simple utility-reranking baseline. URL: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html
- Gneiting and Raftery 2007 Strictly Proper Scoring Rules: probabilistic forecasts should be evaluated as predictive distributions. URL: https://sites.stat.washington.edu/raftery/Research/PDF/Gneiting2007jasa.pdf
- Guo et al. 2017 On Calibration of Modern Neural Networks: post-processing calibration can be a practical way to improve probability estimates, but must be audited separately from classifier accuracy claims. URL: https://proceedings.mlr.press/v70/guo17a/guo17a.pdf
- CheckList behavioral testing: aggregate accuracy can miss targeted behavioral failures, so model tests should specify capability and expected direction. URL: https://aclanthology.org/2020.acl-main.442/
- Contrast Sets: counterfactual/contrast examples should be small, meaningful perturbations that probe local decision boundaries. URL: https://aclanthology.org/2020.findings-emnlp.117/
- Counterfactually Augmented Data: counterfactual revisions should be internally coherent and avoid gratuitous unrelated changes. URL: https://openreview.net/pdf?id=Sklgs0NFvr

## Design principles
Current-data before scale-up; status PASS is not claim allowed; no alpha without after-cost daily evidence; no Flow claim unless Flow beats proxy and the gain survives no-flow/only-flow attribution ablations; no multimodal claim unless news evidence contributes beyond technical baseline.
