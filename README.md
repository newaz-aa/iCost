
# iCost

iCost is a Python library for instance-level cost-sensitive learning, fully compatible with scikit-learn. It extends traditional cost-sensitive classification by dynamically adjusting sample costs based on instance complexity. Multiple strategies have been incorporated into the algorithm, and it works with any scikit-learn classifier that supports sample_weight.

### Requirements:
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/downloads/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-0.24%2B-orange)](https://scikit-learn.org/stable/)
[![numpy](https://img.shields.io/badge/numpy-1.19%2B-ff69b4)](https://numpy.org/)
[![pandas](https://img.shields.io/badge/pandas-1.1%2B-yellow)](https://pandas.pydata.org/)
![Seaborn](https://img.shields.io/badge/Seaborn-Data%20Visualization-blue)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Data%20Visualization-orange)


 

### Key Features:
- Support for any scikit-learn compatible classifier as the base model.
- Multiple strategies for cost-sensitive learning:

    -- ncs → no cost (baseline).
  
    -- org → original sklearn-style cost-sensitive (all minority weighted by imbalance ratio).
  
    -- mst → MST-based linked vs. pure minority categorization.
  
    -- neighbor → neighbor-based categorization with three sub-modes.

- Neighbor-based categorization (5-NN):

    -- Mode 1 → safe, pure, border.
  
    -- Mode 2 → safe, border, outlier.
  
    -- Mode 3 → fine-grained categories g1–g6 with user-defined penalties.

- Utility function: categorize_minority_class for direct analysis of minority-class samples.

  
## Synopsis

The standard weighted classifier applies an increased weight to all the minority class misclassifications in imbalanced classification tasks. This approach is available in the standard implementation of the sklearn library.

However, there is an issue. Should the same weight be applied to all the minority class samples indiscriminately? Some minority class samples are closer to the decision boundary (difficult to identify), while some samples are far way from the border (easy to classify). There are also some instances that are noisy, completely surrounded by instances from the majority class. Now, applying the same higher misclassification cost to all the minority-class samples is unjustifiable. It distorts the decision boundary significantly, resulting in more misclassifications. 

The proposed solution is to apply the cost to only certain samples or apply different costs depending on their level of difficulty. This improves the prediction performance in different imbalanced scenarios.

For more information, please refer to the following paper:

### Paper

arxiv: https://doi.org/10.48550/arXiv.2409.13007 

The paper is currently under review.

## Installation

[![PyPI version](https://img.shields.io/pypi/v/icost?color=blue&label=install%20with%20pip)](https://pypi.org/project/icost/)


```
pip install icost
```


## Usage Example

```
from icost import iCost, categorize_minority_class
from sklearn.svm import SVC

# Example with neighbor-mode cost assignment
clf = iCost(
    base_classifier=SVC(kernel="rbf", probability=True),
    method="neighbor",
    neighbor_mode=2          # Mode 1, 2, or 3
)

clf.fit(X_train, y_train)
print("Test Accuracy:", clf.score(X_test, y_test))

# Example with mode=3 (custom penalties for g1..g6)
clf3 = iCost(
    base_classifier=SVC(),
    method="neighbor",
    neighbor_mode=3,
    neighbor_costs=[1.0, 2.0, 5.0, 5.0, 3.0, 1.0]  # g1..g6
)
clf3.fit(X_train, y_train)
```

### Helper Function

You can analyze minority samples directly with:

```
import pandas as pd
from icost import categorize_minority_class

df = pd.read_csv("your_dataset.csv")
min_idx, groups, opp_counts = categorize_minority_class(
    df,
    minority_label=1,
    mode=1,
    show_summary=True
)
```

### Output:

```
Category summary (minority samples):
  safe: 45
  pure: 28
  border: 62
```

## Structure

```
icost/
├── __init__.py               # Makes icost a package; exposes iCost and helpers
├── __version__.py            # Stores the package version (e.g., 0.1.0)
├── icost.py                  # Main iCost class (methods: ncs, org, mst, neighbor)
├── mst_linked_ind.py         # MST-based helper:
│                             #   - Identifies 'linked' vs 'pure' minority samples
│                             #   - Used for MST variant of iCost
└── categorize_minority_v2.py # Neighbor-based helper:
                              #   - Categorizes minority samples with 5-NN
                              #   - Supports modes (safe, pure, border, outlier, g1–g6)
                              #   - Provides summary statistics
```

### Other files in the repo

 - README.md → Documentation and usage instructions.
 - LICENSE → Project license (MIT by default).
 - pyproject.toml → Build configuration for packaging and PyPI upload.
 - icost_usage_example → tests to check functionality.


## Screenshots

![App Screenshot](https://github.com/newaz-aa/Modified_Cost_Sensitive_Classifier/blob/main/Figures/categorization.png)

![App Screenshot](https://github.com/newaz-aa/Modified_Cost_Sensitive_Classifier/blob/main/Figures/icsot_lr.png)


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
      url={https://arxiv.org/abs/2409.13007}, 
}
```

### License

This project is licensed under the MIT License.

### Note

The work is currently being updated to include additional features, which I plan to incorporate soon. 
