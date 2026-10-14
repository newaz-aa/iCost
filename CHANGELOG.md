# Changelog

## 0.3.0

This release aligns the public package with the revised iCost methodology used in the current manuscript experiments.

### Changed
- Gini-iCost now uses **three-fold cross-fitted complexity estimation** by default, so a training observation is not assigned complexity by a probe tree fitted on that same observation.
- Recommended fixed iCost penalty multipliers are now **(cfo, cfp, cfs, cfb) = (0.10, 0.30, 0.50, 0.70)**, scaled by the imbalance ratio by default.
- Recommended Gini structural defaults are **tau_min=0.80**, **tau_maj=0.20**, `max_depth=3`, and `min_samples_leaf=5`.
- Neighbor categories are now defined by the majority-neighbour ratio, so non-default values of `n_neighbors` are handled consistently.
- Multiclass classification is supported directly through internal one-vs-rest decomposition. The less frequent binary group is identified independently in every OvR subproblem.
- Conventional CS now exposes `cs_multiplier`, giving minority weight `cs_multiplier * IR`.
- Terminology was updated from "noisy" to the more cautious "outlier-like" / "structurally atypical" description.
- Added **Cluster-iCost** as an exploratory extension using separate minority/majority KMeans regions, relative nearest-centre distance, and fixed thresholds `(0.25, 0.50, 0.75)`.
- Corrected scikit-learn estimator mixin order so iCost is recognized as a classifier by modern scikit-learn.
- `decision_function` and `predict_proba` are now exposed conditionally, preventing probability-only classifiers (e.g., decision trees/random forests) from being routed through an unsupported response method during scoring.
- Complexity summaries now include zero-count categories for a stable output schema.

### Compatibility
- `method="gini"` remains an alias for `method="tree"`.
- Legacy `method="org"` remains an alias for conventional CS.
- Old parameter names `tau_pure`, `tau_out`, `tree_max_depth`, `tree_min_samples_leaf`, and `positive_label` remain accepted as aliases.

### Tests
- Added coverage for binary Neighbor-iCost, cross-fitted Gini-iCost, exploratory Cluster-iCost, conventional CS multipliers, multiclass OvR fitting, legacy aliases, sparse input, scikit-learn cloning/classifier tags, and ROC-AUC/AUPR scoring with probability-only classifiers.
