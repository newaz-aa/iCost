# iCost

**iCost** is a scikit-learn-compatible Python package for instance-complexity-aware cost-sensitive learning in imbalanced classification.

Traditional cost-sensitive learning usually assigns the same penalty to every minority-class sample. However, minority samples are not equally difficult to classify. Some are clearly separable, some are safe but near the class boundary, some lie in overlapping regions, and some may be noisy or outlier-like. iCost addresses this limitation by assigning adaptive penalties to minority-class instances according to their estimated learning difficulty.

The package implements two main variants:

* **Neighbor-iCost**: estimates minority-instance complexity using local neighborhood composition.
* **Gini-iCost**: estimates minority-instance complexity using Gini-impurity-based feature-space partitioning with a shallow decision-tree probe.

The framework works with standard classifiers that support `sample_weight`, including Logistic Regression, SVM, Decision Tree, Random Forest, and XGBoost.

---

## Installation

Install from PyPI:

```bash
pip install icost
```

Or install the latest version from GitHub:

```bash
pip install git+https://github.com/newaz-aa/iCost.git
```

---

## Key Features

* Instance-complexity-aware cost-sensitive learning
* Compatible with the scikit-learn estimator interface
* Works with classifiers that support `sample_weight`
* Supports binary imbalanced classification
* Can be extended to multiclass problems using one-vs-rest decomposition
* Does not generate synthetic samples or remove existing samples
* Provides adaptive penalties for minority-class instances based on learning difficulty

---

## Supported Training Modes

| Mode       | Description                                                                                     |
| ---------- | ----------------------------------------------------------------------------------------------- |
| `ncs`      | Non-cost-sensitive baseline. All samples receive equal weight.                                  |
| `cs`       | Conventional cost-sensitive learning. All minority samples receive the same IR-based weight.    |
| `neighbor` | Neighbor-iCost. Minority samples are categorized using local neighborhood composition.          |
| `tree`     | Gini-iCost. Minority samples are weighted using Gini-impurity-based feature-space partitioning. |
| `gini`     | Alias for `tree`.                                                                               |

---

## Default Cost Hierarchy

The default iCost penalties are defined relative to the imbalance ratio, `IR`:

| Minority-instance type | Symbol | Default penalty |
| ---------------------- | -----: | --------------: |
| Outlier-like/noisy     |  `cfo` |     `0.10 × IR` |
| Pure/easy              |  `cfp` |     `0.30 × IR` |
| Safe                   |  `cfs` |     `0.75 × IR` |
| Border/overlapping     |  `cfb` |     `1.00 × IR` |

Thus, the intended hierarchy is:

```text
cfo < cfp < cfs < cfb = IR
```

This allows iCost to emphasize informative boundary samples while reducing unnecessary over-penalization of easy or potentially noisy minority samples.

---

## Basic Usage

### Neighbor-iCost

```python
from icost import iCost
from sklearn.linear_model import LogisticRegression

model = iCost(
    base_classifier=LogisticRegression(max_iter=1000),
    method="neighbor"
)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

### Gini-iCost

```python
from icost import iCost
from xgboost import XGBClassifier

model = iCost(
    base_classifier=XGBClassifier(),
    method="tree",
    tree_max_depth=3,
    tree_min_samples_leaf=5,
    tau_pure=0.80,
    tau_out=0.20
)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

---

## Conventional Cost-Sensitive Learning Baseline

To use standard class-level cost-sensitive learning:

```python
from icost import iCost
from sklearn.svm import SVC

model = iCost(
    base_classifier=SVC(),
    method="cs"
)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

In this mode, all minority-class samples receive the same imbalance-ratio-based penalty.

---

## Custom Cost Values

You can manually set the penalty multipliers:

```python
model = iCost(
    base_classifier=LogisticRegression(max_iter=1000),
    method="neighbor",
    cfo=0.10,
    cfp=0.30,
    cfs=0.75,
    cfb=1.25,
    scale_costs_by_ir=True
)
```

When `scale_costs_by_ir=True`, the values are multiplied by the imbalance ratio.

---

## Example: Multiclass Extension

iCost can be used for multiclass imbalanced classification through one-vs-rest decomposition:

```python
from icost import iCost
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression

base_model = iCost(
    base_classifier=LogisticRegression(max_iter=1000),
    method="neighbor"
)

model = OneVsRestClassifier(base_model)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

## Requirements

- Python >= 3.8
- numpy
- pandas
- scikit-learn


---

## Project Structure

```text
icost/
├── __init__.py
├── __version__.py
├── icost.py                  # Main iCost estimator
└── categorize_minority_v2.py # Helper functions for minority-instance analysis
```

---
---

## Method Overview

### Neighbor-iCost

Neighbor-iCost estimates minority-instance complexity using local neighborhood composition. Minority samples are categorized as pure, safe, border, or outlier-like based on the number of majority-class samples among their nearest neighbors.

![Neighbor-iCost minority-instance categorization](https://github.com/newaz-aa/iCost/blob/main/Figures/neighbor%20categories.png)

### Gini-iCost

Gini-iCost uses a shallow decision-tree probe to partition the feature space. The class distribution and Gini impurity of the leaf containing each minority sample are then used to estimate regional ambiguity and assign adaptive penalties.

![Gini-iCost feature-space partitioning](https://github.com/newaz-aa/iCost/blob/main/Figures/tree_v2.drawio.png)

---



## Research Paper

This package supports the implementation of the following manuscript:

**iCost: A Novel Instance-Complexity-Based Cost-Sensitive Learning Framework**

The manuscript is currently submitted to *Machine Learning with Applications*.

---


---

 


## BibTex Citation
If you plan to use this module, please cite the paper:

```
@misc{newaz2024icostnovelinstancecomplexity,
      title={iCost: A Novel Instance Complexity Based Cost-Sensitive Learning Framework for Imbalanced Classification}, 
      author={Asif Newaz and Asif Ur Rahman Adib and Taskeed Jabid},
      year={2024},
      eprint={2409.13007},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      doi= {https://doi.org/10.48550/arXiv.2409.13007},
      url={https://arxiv.org/abs/2409.13007}, 
}
```


## License

This project is licensed under the MIT License.

---

## Author

**Asif Newaz**  
Assistant Professor  
Department of Electrical and Electronic Engineering  
Islamic University of Technology (IUT), Gazipur, Bangladesh  
Email: [eee.asifnewaz@iut-dhaka.edu](mailto:eee.asifnewaz@iut-dhaka.edu)
