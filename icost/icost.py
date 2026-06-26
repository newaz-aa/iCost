# -*- coding: utf-8 -*-
"""
iCost: instance-complexity-aware cost-sensitive learning for imbalanced classification.

This module implements two iCost variants:

1. Neighbor-iCost:
   Estimates minority-instance complexity using local neighborhood composition.

2. Gini-iCost:
   Estimates minority-instance complexity using Gini-impurity-based
   feature-space partitioning with a shallow decision-tree probe.

The estimator follows the scikit-learn API and works with classifiers that
support the sample_weight argument during fitting.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.neighbors import NearestNeighbors
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.validation import check_is_fitted


class iCost(BaseEstimator, ClassifierMixin):
    """
    Instance-complexity-aware cost-sensitive classifier.

    Parameters
    ----------
    base_classifier : estimator
        Any scikit-learn-compatible classifier that supports sample_weight
        during fitting.

    method : {'ncs', 'cs', 'neighbor', 'tree', 'gini'}, default='cs'
        Training mode.

        - 'ncs'      : non-cost-sensitive baseline; all samples receive weight 1.
        - 'cs'       : conventional cost-sensitive learning; all minority samples
                       receive the same imbalance-ratio-based weight.
        - 'neighbor' : Neighbor-iCost using local neighborhood composition.
        - 'tree'     : Gini-iCost using decision-tree-based feature-space partitioning.
        - 'gini'     : alias for 'tree'.

        The legacy alias 'org' is also accepted and treated as 'cs'.

    n_neighbors : int, default=5
        Number of nearest neighbors used by Neighbor-iCost.

    cfo : float or None, default=None
        Cost multiplier for outlier-like/noisy minority samples.
        If None, the default value 0.10 is used.

    cfp : float or None, default=None
        Cost multiplier for pure/easy minority samples.
        If None, the default value 0.30 is used.

    cfs : float or None, default=None
        Cost multiplier for safe minority samples.
        If None, the default value 0.75 is used.

    cfb : float or None, default=None
        Cost multiplier for border/overlapping minority samples.
        If None, the default value 1.00 is used.

    cost_factor : dict or None, default=None
        Optional dictionary for setting cost multipliers, e.g.
        {'cfo': 0.10, 'cfp': 0.30, 'cfs': 0.75, 'cfb': 1.00}.
        Values in this dictionary override the corresponding cfo/cfp/cfs/cfb
        arguments.

    scale_costs_by_ir : bool, default=True
        If True, cfo/cfp/cfs/cfb are multiplied by the imbalance ratio.
        If False, cfo/cfp/cfs/cfb are used directly as sample weights.

    tau_pure : float, default=0.80
        Threshold for identifying minority-dominated leaves in Gini-iCost.

    tau_out : float, default=0.20
        Threshold for identifying majority-dominated leaves in Gini-iCost.

    tree_max_depth : int or None, default=3
        Maximum depth of the decision-tree probe used by Gini-iCost.

    tree_min_samples_leaf : int, default=5
        Minimum number of samples required in each leaf of the decision-tree probe.

    positive_label : int, str, or None, default=None
        Label to be treated as the minority/positive class. If None, the rarest
        class in the training labels is used.

    random_state : int or None, default=None
        Random state used by the decision-tree probe.
    """

    def __init__(
        self,
        base_classifier,
        method="cs",
        n_neighbors=5,
        cfo=None,
        cfp=None,
        cfs=None,
        cfb=None,
        cost_factor=None,
        scale_costs_by_ir=True,
        tau_pure=0.80,
        tau_out=0.20,
        tree_max_depth=3,
        tree_min_samples_leaf=5,
        positive_label=None,
        random_state=None,
    ):
        self.base_classifier = base_classifier
        self.method = method
        self.n_neighbors = n_neighbors
        self.cfo = cfo
        self.cfp = cfp
        self.cfs = cfs
        self.cfb = cfb
        self.cost_factor = cost_factor
        self.scale_costs_by_ir = scale_costs_by_ir
        self.tau_pure = tau_pure
        self.tau_out = tau_out
        self.tree_max_depth = tree_max_depth
        self.tree_min_samples_leaf = tree_min_samples_leaf
        self.positive_label = positive_label
        self.random_state = random_state

    @staticmethod
    def _as_1d(y):
        y = np.asarray(y)
        if y.ndim > 1:
            y = y.ravel()
        return y

    @staticmethod
    def _to_numpy(X):
        if isinstance(X, (pd.DataFrame, pd.Series)):
            return X.to_numpy()
        return np.asarray(X)

    def _normalize_method(self):
        method = str(self.method).lower()

        if method == "org":
            method = "cs"

        if method == "gini":
            method = "tree"

        valid_methods = {"ncs", "cs", "neighbor", "tree"}
        if method not in valid_methods:
            raise ValueError(
                "method must be one of {'ncs', 'cs', 'neighbor', 'tree', 'gini'}."
            )

        return method

    def _resolve_minority_label(self, y):
        classes, counts = np.unique(y, return_counts=True)

        if len(classes) < 2:
            raise ValueError("iCost requires at least two classes in y.")

        if self.positive_label is not None:
            if self.positive_label not in classes:
                raise ValueError("positive_label was not found in y.")
            return self.positive_label

        return classes[np.argmin(counts)]

    def _calculate_imbalance_ratio(self, y, minority_label):
        minority_count = np.sum(y == minority_label)
        majority_count = len(y) - minority_count

        if minority_count == 0:
            raise ValueError("No minority samples found.")

        if majority_count == 0:
            raise ValueError("No majority samples found.")

        return majority_count / minority_count

    def _resolve_costs(self, ir):
        """
        Resolve iCost penalty values.

        Default recommended mild configuration:
            cfo = 0.10 * IR
            cfp = 0.30 * IR
            cfs = 0.75 * IR
            cfb = 1.00 * IR
        """
        defaults = {
            "cfo": 0.10,
            "cfp": 0.30,
            "cfs": 0.75,
            "cfb": 1.00,
        }

        direct_values = {
            "cfo": self.cfo,
            "cfp": self.cfp,
            "cfs": self.cfs,
            "cfb": self.cfb,
        }

        cost_factor = self.cost_factor or {}

        resolved = {}
        for key, default_value in defaults.items():
            value = cost_factor.get(key, direct_values[key])

            if value is None:
                value = default_value

            value = float(value)

            if value < 0:
                raise ValueError(f"{key} must be non-negative.")

            resolved[key] = value * ir if self.scale_costs_by_ir else value

        return resolved

    def compute_sample_weight(self, X, y):
        """
        Compute sample weights according to the selected iCost mode.
        """
        X_np = self._to_numpy(X)
        y = self._as_1d(y)

        if X_np.shape[0] != len(y):
            raise ValueError("X and y have inconsistent numbers of samples.")

        method = self._normalize_method()
        minority_label = self._resolve_minority_label(y)
        ir = self._calculate_imbalance_ratio(y, minority_label)
        costs = self._resolve_costs(ir)

        weights = np.ones(len(y), dtype=float)
        minority_idx = np.where(y == minority_label)[0]

        if method == "ncs":
            return weights

        if method == "cs":
            weights[minority_idx] = ir
            return weights

        if method == "neighbor":
            return self._neighbor_weights(X_np, y, minority_idx, minority_label, costs)

        if method == "tree":
            return self._tree_weights(X_np, y, minority_idx, minority_label, costs)

        raise RuntimeError("Unexpected method encountered.")

    def _neighbor_weights(self, X, y, minority_idx, minority_label, costs):
        weights = np.ones(len(y), dtype=float)

        if len(y) < 2:
            raise ValueError("At least two samples are required for Neighbor-iCost.")

        n_neighbors = int(self.n_neighbors)
        if n_neighbors < 1:
            raise ValueError("n_neighbors must be at least 1.")

        k = min(n_neighbors + 1, len(y))

        knn = NearestNeighbors(n_neighbors=k)
        knn.fit(X)

        _, indices = knn.kneighbors(X[minority_idx])

        for row, sample_idx in zip(indices, minority_idx):
            neighbors = [idx for idx in row if idx != sample_idx][:n_neighbors]

            if not neighbors:
                continue

            majority_count = int(np.sum(y[neighbors] != minority_label))

            if majority_count == 0:
                weights[sample_idx] = costs["cfp"]      # pure/easy
            elif majority_count in (1, 2):
                weights[sample_idx] = costs["cfs"]      # safe
            elif majority_count in (3, 4):
                weights[sample_idx] = costs["cfb"]      # border/overlap
            else:
                weights[sample_idx] = costs["cfo"]      # outlier-like/noisy

        return weights

    def _tree_weights(self, X, y, minority_idx, minority_label, costs):
        weights = np.ones(len(y), dtype=float)

        if not 0 <= self.tau_out <= 1:
            raise ValueError("tau_out must be in the range [0, 1].")

        if not 0 <= self.tau_pure <= 1:
            raise ValueError("tau_pure must be in the range [0, 1].")

        if self.tau_out >= self.tau_pure:
            raise ValueError("tau_out must be smaller than tau_pure.")

        probe_tree = DecisionTreeClassifier(
            criterion="gini",
            max_depth=self.tree_max_depth,
            min_samples_leaf=self.tree_min_samples_leaf,
            random_state=self.random_state,
        )

        probe_tree.fit(X, y)
        leaves = probe_tree.apply(X)

        for sample_idx in minority_idx:
            leaf_id = leaves[sample_idx]
            in_leaf = leaves == leaf_id

            n_min = np.sum((y == minority_label) & in_leaf)
            n_maj = np.sum((y != minority_label) & in_leaf)
            total = n_min + n_maj

            if total == 0:
                weights[sample_idx] = costs["cfp"]
                continue

            p_min = n_min / total
            p_maj = 1.0 - p_min

            gini = 1.0 - (p_min ** 2) - (p_maj ** 2)
            ambiguity = 2.0 * gini

            if p_min >= self.tau_pure:
                weights[sample_idx] = costs["cfp"]
            elif p_min <= self.tau_out:
                weights[sample_idx] = costs["cfo"]
            else:
                weights[sample_idx] = costs["cfs"] + (costs["cfb"] - costs["cfs"]) * ambiguity

        return weights

    def apply_cost(self, X, y):
        """
        Backward-compatible alias for compute_sample_weight.
        """
        return self.compute_sample_weight(X, y)

    def calculate_imbalance_ratio(self, y):
        """
        Public helper to calculate the imbalance ratio.
        """
        y = self._as_1d(y)
        minority_label = self._resolve_minority_label(y)
        return self._calculate_imbalance_ratio(y, minority_label)

    def fit(self, X, y):
        """
        Fit the base classifier using the selected iCost sample weights.
        """
        X_np = self._to_numpy(X)
        y = self._as_1d(y)

        if X_np.shape[0] != len(y):
            raise ValueError("X and y have inconsistent numbers of samples.")

        self.classes_ = np.unique(y)
        self.minority_label_ = self._resolve_minority_label(y)
        self.imbalance_ratio_ = self._calculate_imbalance_ratio(y, self.minority_label_)
        self.sample_weight_ = self.compute_sample_weight(X_np, y)

        self.estimator_ = clone(self.base_classifier)

        try:
            self.estimator_.fit(X, y, sample_weight=self.sample_weight_)
        except TypeError as exc:
            method = self._normalize_method()

            if method == "ncs":
                self.estimator_.fit(X, y)
            else:
                raise TypeError(
                    f"{self.base_classifier.__class__.__name__} does not support "
                    "sample_weight. Please use a classifier that supports sample weighting."
                ) from exc

        return self

    def predict(self, X):
        """
        Predict class labels for samples in X.
        """
        check_is_fitted(self, "estimator_")
        return self.estimator_.predict(X)

    def predict_proba(self, X):
        """
        Predict class probabilities for samples in X, if supported by the base classifier.
        """
        check_is_fitted(self, "estimator_")

        if hasattr(self.estimator_, "predict_proba"):
            return self.estimator_.predict_proba(X)

        raise AttributeError(
            f"{self.estimator_.__class__.__name__} does not support predict_proba."
        )

    def decision_function(self, X):
        """
        Compute decision scores for samples in X, if supported by the base classifier.
        """
        check_is_fitted(self, "estimator_")

        if hasattr(self.estimator_, "decision_function"):
            return self.estimator_.decision_function(X)

        raise AttributeError(
            f"{self.estimator_.__class__.__name__} does not support decision_function."
        )

    def score(self, X, y):
        """
        Return the mean accuracy on the given test data and labels.
        """
        check_is_fitted(self, "estimator_")
        return self.estimator_.score(X, y)
