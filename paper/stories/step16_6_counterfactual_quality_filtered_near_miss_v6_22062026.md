# Step 16.6 Story - Counterfactual Quality-Filtered Near Miss

Step 16 originally looked like a broad counterfactual-faithfulness failure: pass rate `0.3257`, no-change `0.3657`. The deeper issue was that many tasks were mixed-polarity or had placeholder counterfactual contexts. Step 16.6 applied a contrast-set quality filter: keep tasks where expected polarity is dominant and actually removed, repair placeholder context text, and round-robin the selected tasks across types.

The quality-controlled rerun is much stronger: schema `0.9974`, no-change `0.2604`, and the news removal gates pass (`remove_positive=0.50`, `remove_negative=0.7143`). The overall pass rate is still `0.4844`, just under the `0.50` claim gate, driven mainly by `neutralize_negative_evidence=0.0952`.

Paper angle: this is a useful negative-result narrative. The model is sensitive to removal counterfactuals but not to negative-evidence neutralization. That suggests the next method should train on pairwise counterfactual deltas or explicit evidence-sensitivity preferences rather than merely improving the evaluator.
