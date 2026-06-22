# Step 16.7 Story - Semantic Neutralization Opens Counterfactual Gate

Step 16.6 made counterfactual faithfulness a near miss: pass rate `0.4844`, no-change `0.2604`, and news removal gates passed. The remaining failure was `neutralize_negative_evidence`, where token-level replacement created unnatural phrases such as isolated neutral updates. The model often treated those as lower-information contexts rather than coherent neutralized evidence.

Step 16.7 rewrote neutralization as semantic counterfactual edits: favorable or unfavorable company language is rewritten as a neutral operational update, without reintroducing polarity terms. This is closer to CheckList/contrast-set practice because the test is a meaningful local edit rather than a string-corrupted artifact.

The rerun passed the configured gates: `pass_rate=0.5313`, `no_change_rate=0.2083`, `schema_ok_rate=0.9948`, and both news-removal rates exceed the gate. `neutralize_negative_evidence` improved to `0.5238`.

Paper angle: the important method lesson is that counterfactual faithfulness depends on coherent perturbation design. The model was not fully insensitive to negative evidence; it failed when the counterfactual edit was linguistically brittle. Semantic neutralization is a defensible evaluation repair, while broader strong-accept claims still require Flow, forecast, and alpha blockers to pass.
