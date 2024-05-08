# Modified_Cost_Sensitive_SVM_Classifier
The standard weighted SVM classifier applies the increased weight to all the minority class samples in imbalanced classification tasks. This approach is available in the SVC implementation of the sklearn library.

However, there is an issue. Should the same weight be applied to all the minority class samples indiscriminately? Some minority class samples are closer to the decision boundary (difficult to identify), while some samples are far way from the border (easy to classify). Now, applying the higher misclassification cost to all the minority-class samples might result in more misclassifications of the majority-class samples. A possible solution is to apply the cost to only certain samples or apply different cost depending on their level of difficulty.

The code here first divides the minority class samples in four categories: Safe, Borderline, Rare, and Outlier. Then, it applies the given misclassification cost (input) to all minority class samples except the ones that are marked as Safe. The flag parameter allows to use the class as a standard cost-sensitive SVM or modified cost-sensitive SVM, depending on user requirement.

Applying the proposed approach on around 60 imbalanced datasets have shown about 1-4% improvement in MCC score. The approach improves the specificity without reducing the sensitivity score.
