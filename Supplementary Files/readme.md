# Supplementary Materials

This folder contains the supplementary materials accompanying the manuscript:

**iCost: A Novel Instance-Complexity-Based Cost-Sensitive Learning Framework**

The supplementary files provide detailed dataset information, complete experimental results, statistical analyses, sensitivity studies, mechanism controls, and additional analyses supporting the results reported in the manuscript.

## Supplementary Files

| File | Description |
|---|---|
| [S1_Dataset_Details.xlsx](S1_Dataset_Details.xlsx) | Characteristics of the binary, multiclass, large-scale, and synthetic datasets, including the 15 representative datasets used in sensitivity and mechanism analyses. |
| [S2_Binary_Results.xlsx](S2_Binary_Results.xlsx) | Tuned results for the 65 binary benchmark datasets. |
| [S3_Large_Scale_Results.xlsx](S3_Large_Scale_Results.xlsx) | Performance, runtime, and error trade-off results for the large-scale application-oriented datasets. |
| [S4_Default_Cost_Results.xlsx](S4_Default_Cost_Results.xlsx) | Results obtained using the default cost configuration, together with comparisons against dataset-specific tuned settings. |
| [S5_Statistical_Analysis.xlsx](S5_Statistical_Analysis.xlsx) | Friedman tests, Wilcoxon signed-rank tests, Holm-adjusted p-values, effect sizes, bootstrap confidence intervals, and dataset-level scores. |
| [S6_Multiclass_Results.xlsx](S6_Multiclass_Results.xlsx) | Detailed results for the nine multiclass benchmark datasets. |
| [S7_Resampling_Comparison.xlsx](S7_Resampling_Comparison.xlsx) | Detailed comparison of iCost with the evaluated oversampling, undersampling, and hybrid resampling methods. |
| [S8_Recent_CSL_Comparison.xlsx](S8_Recent_CSL_Comparison.xlsx) | Detailed comparison with recent cost-sensitive learning methods, including CSLKim and TSBQT. |
| [S9_Computational_Time.xlsx](S9_Computational_Time.xlsx) | Computational-time breakdowns by method, classifier, and dataset, including complexity-estimation, training, prediction, and tuning costs. |
| [S10_Complexity_Distributions.xlsx](S10_Complexity_Distributions.xlsx) | Minority-instance complexity distributions under the controlled experimental settings. |
| [S11_Parameter_Sensitivity.xlsx](S11_Parameter_Sensitivity.xlsx) | Sensitivity results for Neighbor-iCost neighborhood size and the structural parameters of Gini-iCost. |
| [S12_Mechanism_Robustness_Controls.xlsx](S12_Mechanism_Robustness_Controls.xlsx) | Mean-matched and randomly permuted cost controls used to examine the effect of complexity-dependent cost assignment. |
| [S13_Controlled_Data_Complexity.xlsx](S13_Controlled_Data_Complexity.xlsx) | Synthetic-data performance results, structural complexity measures, correlations, NCS references, and FP-FN trade-off efficiency results under varying data difficulty. |
| [S14_k_sensitivity_category_distribution.pdf](S14_k_sensitivity_category_distribution.pdf) | Effect of neighborhood size `k` on the distribution of Neighbor-iCost minority-instance complexity categories. |
| [S15_Response_of_Complexity_Estimators.pdf](S15_Response_of_Complexity_Estimators.pdf) | Additional analysis showing how the proposed complexity estimators respond to changes in data difficulty. |

## Notes

The main manuscript reports the principal findings and summary results, while these supplementary files provide the corresponding detailed dataset-wise, classifier-wise, and experimental outputs.

All supplementary results should be interpreted together with the experimental design, validation protocol, performance measures, and statistical procedures described in the manuscript.

