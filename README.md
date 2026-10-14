# iCost

**iCost** is a scikit-learn-compatible Python package for instance-complexity-aware cost-sensitive learning in imbalanced classification.

Conventional cost-sensitive learning normally assigns one class-level penalty to all minority samples. iCost instead derives sample-specific penalties from structural instance complexity. The package implements two primary variants and one exploratory extension:

- **Neighbor-iCost**: local neighbourhood composition identifies pure, safe, border, and outlier-like minority instances.
- **Gini-iCost**: cross-fitted shallow decision-tree probes estimate regional ambiguity from leaf class composition and normalized Gini impurity.
- **Cluster-iCost (exploratory)**: class-specific KMeans regions provide a broader structural view by comparing each minority sample with its nearest minority and majority cluster centres.

The framework works with classifiers that support `sample_weight`, including Logistic Regression, SVM, Decision Tree, Random Forest, and XGBoost.

## Installation

```bash
pip install icost
```

Latest GitHub version:

```bash
pip install git+https://github.com/newaz-aa/iCost.git
```

## Training modes

| Mode | Description |
|---|---|
| `ncs` | Non-cost-sensitive baseline; all observations receive weight 1. |
| `cs` | Conventional class-level CSL; the minority group receives `cs_multiplier * IR`. |
| `neighbor` | Neighbor-iCost. |
| `tree` | Gini-iCost with cross-fitted probe-tree complexity. |
| `gini` | Alias for `tree`. |
| `cluster` | Exploratory Cluster-iCost using class-specific KMeans regions. |

## Recommended default configuration

The default iCost penalty multipliers are scaled by the imbalance ratio (`IR`):

| Minority-instance type | Parameter | Default |
|---|---:|---:|
| Structurally atypical / outlier-like | `cfo` | `0.10 * IR` |
| Pure / easy | `cfp` | `0.30 * IR` |
| Safe | `cfs` | `0.50 * IR` |
| Border / strongly ambiguous | `cfb` | `0.70 * IR` |

Neighbor-iCost uses `n_neighbors=5` by default.

Gini-iCost uses:

```text
crossfit_folds = 3
max_depth = 3
min_samples_leaf = 5
tau_min = 0.80
tau_maj = 0.20
```

## Basic usage

### Neighbor-iCost

```python
from icost import iCost
from sklearn.linear_model import LogisticRegression

model = iCost(
    base_classifier=LogisticRegression(max_iter=1000),
    method="neighbor",
    random_state=42,
)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

### Gini-iCost

```python
from icost import iCost
from sklearn.linear_model import LogisticRegression

model = iCost(
    base_classifier=LogisticRegression(max_iter=1000),
    method="tree",
    crossfit_folds=3,
    max_depth=3,
    min_samples_leaf=5,
    tau_min=0.80,
    tau_maj=0.20,
    random_state=42,
)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```


## Exploratory Cluster-iCost

Cluster-iCost is included as an exploratory extension. KMeans is fitted separately to the minority and majority groups. For each minority sample, the relative distance score is

```text
r = d_min / (d_min + d_maj)
```

where `d_min` and `d_maj` are distances to the nearest minority and majority cluster centres. The default fixed exploratory thresholds `(0.25, 0.50, 0.75)` map the score to pure, safe, border, and outlier-like categories. The same default iCost penalties `(0.10, 0.30, 0.50, 0.70) * IR` are then applied.

```python
from icost import iCost
from sklearn.linear_model import LogisticRegression

model = iCost(
    base_classifier=LogisticRegression(max_iter=1000),
    method="cluster",
    n_clusters=2,
    random_state=42,
)
model.fit(X_train, y_train)

# Relative cluster-distance scores for minority training samples
cluster_scores = model.cluster_ratio_by_index_
```

The cluster formulation is currently exploratory and is provided to support further evaluation rather than as a replacement for Neighbor-iCost or Gini-iCost.

## Conventional cost-sensitive baseline

```python
model = iCost(
    base_classifier=LogisticRegression(max_iter=1000),
    method="cs",
    cs_multiplier=1.0,
)
```

The minority weight is `cs_multiplier * IR`. This parameter can be tuned with standard scikit-learn tools.

## Custom iCost penalties

```python
model = iCost(
    base_classifier=LogisticRegression(max_iter=1000),
    method="neighbor",
    cfo=0.10,
    cfp=0.30,
    cfs=0.50,
    cfb=0.70,
    scale_costs_by_ir=True,
)
```

## Multiclass classification

iCost 0.3.0 directly supports multiclass targets through one-vs-rest decomposition. For each class-vs-rest subproblem, the less frequent of the two binary groups is treated as the minority group and receives the selected weighting strategy.

```python
from icost import iCost
from sklearn.linear_model import LogisticRegression

model = iCost(
    base_classifier=LogisticRegression(max_iter=1000),
    method="tree",
    random_state=42,
)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

The older explicit wrapper remains compatible:

```python
from sklearn.multiclass import OneVsRestClassifier

binary_icost = iCost(
    base_classifier=LogisticRegression(max_iter=1000),
    method="neighbor",
)
model = OneVsRestClassifier(binary_icost)
model.fit(X_train, y_train)
```

## Inspecting weights and complexity

Binary fitting stores:

```python
model.sample_weight_
model.minority_label_
model.imbalance_ratio_
model.complexity_summary_
model.cluster_ratio_by_index_  # populated for method="cluster"
```

For multiclass fitting, OvR-specific information is available through:

```python
model.sample_weight_by_class_
model.minority_group_by_class_
model.imbalance_ratio_by_class_
model.complexity_summary_by_class_
```

Weights can also be generated directly for a binary target:

```python
weights = model.compute_sample_weight(X_train, y_train)
```

and for a multiclass target:

```python
weight_dict = model.compute_ovr_sample_weights(X_train, y_train)
```

## Preprocessing

iCost's complexity estimators operate in feature space. Categorical variables should therefore be encoded before fitting. Scaling numeric variables is recommended when using Neighbor-iCost or Cluster-iCost because both rely on Euclidean distances. Preprocessing should be fitted only on the training partition in a validation pipeline.

## Backward compatibility with 0.2.x

The following aliases remain supported:

| 0.2.x name | 0.3.0 canonical name |
|---|---|
| `tau_pure` | `tau_min` |
| `tau_out` | `tau_maj` |
| `tree_max_depth` | `max_depth` |
| `tree_min_samples_leaf` | `min_samples_leaf` |
| `positive_label` | `minority_label` |
| `method="gini"` | `method="tree"` |
| `method="org"` | `method="cs"` |

## Requirements

- Python >= 3.8
- NumPy
- pandas
- SciPy
- scikit-learn

## Research paper

This package supports the implementation of the manuscript:

**iCost: A Novel Instance-Complexity-Based Cost-Sensitive Learning Framework**

The manuscript is currently under revision for *Machine Learning with Applications*.

### Existing arXiv citation

```bibtex
@misc{newaz2024icostnovelinstancecomplexity,
      title={iCost: A Novel Instance Complexity Based Cost-Sensitive Learning Framework for Imbalanced Classification},
      author={Asif Newaz and Asif Ur Rahman Adib and Taskeed Jabid},
      year={2024},
      eprint={2409.13007},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      doi={10.48550/arXiv.2409.13007},
      url={https://arxiv.org/abs/2409.13007}
}
```

## License

MIT License.

## Author

**Asif Newaz**  
Assistant Professor  
Department of Electrical and Electronic Engineering  
Islamic University of Technology (IUT), Gazipur, Bangladesh  
Email: eee.asifnewaz@iut-dhaka.edu
