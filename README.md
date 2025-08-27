
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

    ncs → no cost (baseline).
    org → original sklearn-style cost-sensitive (all minority weighted by imbalance ratio).
    mst → MST-based linked vs. pure minority categorization.
    neighbor → neighbor-based categorization with three sub-modes.

- Neighbor-based categorization (5-NN):

    Mode 1 → safe, pure, border.
    Mode 2 → safe, border, outlier.
    Mode 3 → fine-grained categories g1–g6 with user-defined penalties.

- Utility function: categorize_minority_class for direct analysis of minority-class samples.

  
## Synopsis

The standard weighted classifier applies an increased weight to all the minority class misclassifications in imbalanced classification tasks. This approach is available in the standard implementation of the sklearn library.

However, there is an issue. Should the same weight be applied to all the minority class samples indiscriminately? Some minority class samples are closer to the decision boundary (difficult to identify), while some samples are far way from the border (easy to classify). There are also some instances that are noisy, completely surrounded by instances from the majority class. Now, applying the same higher misclassification cost to all the minority-class samples is unjustifiable. It distorts the decision boundary significantly, resulting in more misclassifications. 

The proposed solution is to apply the cost to only certain samples or apply different costs depending on their level of difficulty. This improves the prediction performance in different imbalanced scenarios.

For more information, please refer to the following paper:

### Paper

arxiv: https://doi.org/10.48550/arXiv.2409.13007 

The paper is currently under review.


## Usage Example

* icost.py module implements the proposed approach. 

* icost_test.ipynb file shows the implementation of the algorithm in several datasets.
  
* icost_figure.ipynb file contains codes for performance visualization.

* icost_mst.py contains the code to implement the iCost algorithm using instance complexity based on the Minimum Spanning Tree (MST).
  
* instance_complexity_mst.py file contains the code to obtain instance classification based on MST. 

### Input arguments
There are three input parameters.
* flag 

```
flag = 0 => all minority samples - cost_factor = 1 [original classifier]
flag = 1 => all minority samples = cost_factor = IR [original CS classifier]
flag = 2 => cost applied based on minority-class instance categories
```

* base_classifier

```
A classifier instance such as SVM or RF. Should support sample_weight parameter.
```

* cost factor (cfp, cfs, cfb)
```
default values are set for these three types of instance categories. 
Modify/tune the parameters for improved performance.
```
## Screenshots

![App Screenshot](https://github.com/newaz-aa/Modified_Cost_Sensitive_Classifier/blob/main/Figures/categorization.png)

![App Screenshot](https://github.com/newaz-aa/Modified_Cost_Sensitive_Classifier/blob/main/Figures/icsot_lr.png)


## BibTex Citation
If you plan to use this module, please consider referring to the following paper:

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
# Note

The work is currently being updated to include additional features which I plan to incorporate soon. 
