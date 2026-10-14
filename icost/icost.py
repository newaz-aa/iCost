# -*- coding: utf-8 -*-
"""
iCost: instance-complexity-aware cost-sensitive learning for imbalanced classification.

This module implements two primary iCost variants plus an exploratory extension:

1. Neighbor-iCost
   Estimates minority-instance complexity from local neighbourhood composition.

2. Gini-iCost
   Estimates minority-instance complexity from cross-fitted, shallow
   decision-tree partitions and leaf-level Gini impurity.

3. Cluster-iCost (exploratory)
   Estimates minority-instance complexity from relative distance to multiple
   class-specific KMeans regions.

The estimator follows the scikit-learn API and works with classifiers that
support ``sample_weight`` during fitting. Multiclass problems are handled
internally with one-vs-rest (OvR) decomposition.
"""

from __future__ import annotations

import warnings
from collections import Counter
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.metrics import pairwise_distances
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.metaestimators import available_if
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_array, check_X_y, check_is_fitted, has_fit_parameter

try:
    from sklearn.utils.validation import validate_data as _sk_validate_data
except ImportError:  # scikit-learn < 1.6
    _sk_validate_data = None


_DEFAULT_COSTS = {
    "cfo": 0.10,
    "cfp": 0.30,
    "cfs": 0.50,
    "cfb": 0.70,
}
_DEFAULT_TAU_MIN = 0.80
_DEFAULT_TAU_MAJ = 0.20
_DEFAULT_MAX_DEPTH = 3
_DEFAULT_MIN_SAMPLES_LEAF = 5
_DEFAULT_CROSSFIT_FOLDS = 3
_DEFAULT_N_CLUSTERS = 2
_DEFAULT_CLUSTER_THRESHOLDS = (0.25, 0.50, 0.75)


class iCost(ClassifierMixin, BaseEstimator):
    """Instance-complexity-aware cost-sensitive classifier.

    Parameters
    ----------
    base_classifier : estimator
        Any scikit-learn-compatible classifier. Cost-sensitive modes require
        the estimator's ``fit`` method to accept ``sample_weight``.

    method : {'ncs', 'cs', 'neighbor', 'tree', 'gini', 'cluster'}, default='cs'
        Learning mode.

        - ``'ncs'``: non-cost-sensitive baseline.
        - ``'cs'``: conventional class-level cost-sensitive learning.
        - ``'neighbor'``: Neighbor-iCost.
        - ``'tree'``: Gini-iCost.
        - ``'gini'``: alias for ``'tree'``.
        - ``'cluster'``: exploratory Cluster-iCost using class-specific KMeans regions.

        The legacy alias ``'org'`` is accepted and mapped to ``'cs'``.

    n_neighbors : int, default=5
        Number of nearest neighbours used by Neighbor-iCost.

    cfo, cfp, cfs, cfb : float or None, default=None
        Relative penalty multipliers for structurally atypical, pure, safe,
        and border minority instances, respectively. ``None`` uses the
        recommended defaults ``(0.10, 0.30, 0.50, 0.70)``. With
        ``scale_costs_by_ir=True`` these values are multiplied by the
        imbalance ratio.

    cost_factor : dict or None, default=None
        Backward-compatible dictionary override containing any of
        ``{'cfo', 'cfp', 'cfs', 'cfb'}``.

    scale_costs_by_ir : bool, default=True
        If True, iCost penalty multipliers are scaled by the imbalance ratio.

    cs_multiplier : float, default=1.0
        Multiplier applied to the conventional CS minority weight. The final
        CS minority weight is ``cs_multiplier * IR``.

    tau_min : float or None, default=None
        Threshold for minority-dominated probe-tree leaves. ``None`` uses 0.80.

    tau_maj : float or None, default=None
        Threshold for majority-dominated probe-tree leaves. ``None`` uses 0.20.

    max_depth : int or None, default=None
        Maximum depth of the Gini probe tree. ``None`` uses 3 unless the legacy
        alias ``tree_max_depth`` is supplied.

    min_samples_leaf : int or None, default=None
        Minimum leaf size of the Gini probe tree. ``None`` uses 5 unless the
        legacy alias ``tree_min_samples_leaf`` is supplied.

    crossfit_folds : int, default=3
        Number of stratified folds used for cross-fitted Gini complexity
        estimation. If a binary subproblem contains fewer minority samples than
        this value, the number of folds is reduced to the largest valid value
        (minimum 2) and a warning is emitted.

    n_clusters : int, default=2
        Number of KMeans regions fitted separately within the minority and
        majority groups for the exploratory Cluster-iCost variant.

    cluster_thresholds : tuple of three floats, default=(0.25, 0.50, 0.75)
        Fixed exploratory thresholds applied to the relative cluster-distance
        score. The intervals define pure, safe, border, and outlier-like
        minority categories, respectively.

    minority_label : object or None, default=None
        Optional explicit minority label for binary classification. By default,
        the less frequent class is used. This option is not used for multiclass
        OvR subproblems, where the less frequent binary group is identified
        independently for each class.

    random_state : int or None, default=None
        Random state used by cross-fitting and probe trees.

    positive_label : object or None, default=None
        Deprecated alias for ``minority_label`` retained for backward
        compatibility.

    tau_pure, tau_out, tree_max_depth, tree_min_samples_leaf : optional
        Deprecated aliases retained for compatibility with iCost 0.2.x.
        ``tau_pure`` maps to ``tau_min`` and ``tau_out`` maps to ``tau_maj``.
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
        cs_multiplier=1.0,
        tau_min=None,
        tau_maj=None,
        max_depth=None,
        min_samples_leaf=None,
        crossfit_folds=_DEFAULT_CROSSFIT_FOLDS,
        n_clusters=_DEFAULT_N_CLUSTERS,
        cluster_thresholds=_DEFAULT_CLUSTER_THRESHOLDS,
        minority_label=None,
        random_state=None,
        positive_label=None,
        tau_pure=None,
        tau_out=None,
        tree_max_depth=None,
        tree_min_samples_leaf=None,
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
        self.cs_multiplier = cs_multiplier
        self.tau_min = tau_min
        self.tau_maj = tau_maj
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.crossfit_folds = crossfit_folds
        self.n_clusters = n_clusters
        self.cluster_thresholds = cluster_thresholds
        self.minority_label = minority_label
        self.random_state = random_state

        # Backward-compatible parameter aliases from 0.2.x.
        self.positive_label = positive_label
        self.tau_pure = tau_pure
        self.tau_out = tau_out
        self.tree_max_depth = tree_max_depth
        self.tree_min_samples_leaf = tree_min_samples_leaf

    def __sklearn_tags__(self):
        """Advertise classifier/sparse-input support to modern scikit-learn."""
        parent = getattr(super(), "__sklearn_tags__", None)
        if parent is None:  # compatibility with older scikit-learn
            return self._more_tags()
        tags = parent()
        if hasattr(tags, "input_tags"):
            tags.input_tags.sparse = True
        return tags

    def _more_tags(self):
        """Compatibility tags for older scikit-learn releases."""
        return {"X_types": ["2darray", "sparse"]}

    # ------------------------------------------------------------------
    # Validation / parameter resolution
    # ------------------------------------------------------------------
    @staticmethod
    def _as_1d(y):
        y = np.asarray(y)
        if y.ndim > 1:
            y = y.ravel()
        return y

    @staticmethod
    def _complexity_X(X):
        """Return a row-sliceable numeric matrix without densifying sparse X."""
        if sparse.issparse(X):
            return X.tocsr()
        if isinstance(X, (pd.DataFrame, pd.Series)):
            return X.to_numpy()
        return np.asarray(X)

    @staticmethod
    def _n_samples(X):
        if not hasattr(X, "shape") or len(X.shape) < 1:
            raise ValueError("X must be a 2D array-like object.")
        return int(X.shape[0])

    def _normalize_method(self):
        method = str(self.method).lower()
        if method == "org":
            method = "cs"
        if method == "gini":
            method = "tree"
        valid = {"ncs", "cs", "neighbor", "tree", "cluster"}
        if method not in valid:
            raise ValueError(
                "method must be one of {'ncs', 'cs', 'neighbor', 'tree', 'gini', 'cluster'}."
            )
        return method

    def _resolved_minority_override(self):
        if self.minority_label is not None and self.positive_label is not None:
            if self.minority_label != self.positive_label:
                raise ValueError(
                    "minority_label and deprecated positive_label were both supplied "
                    "with different values."
                )
        return self.minority_label if self.minority_label is not None else self.positive_label

    @staticmethod
    def _resolve_minority_label(y, override=None):
        classes, counts = np.unique(y, return_counts=True)
        if len(classes) != 2:
            raise ValueError("Binary weighting requires exactly two classes.")
        if override is not None:
            if override not in classes:
                raise ValueError("The requested minority label was not found in y.")
            return override
        return classes[np.argmin(counts)]

    @staticmethod
    def _calculate_imbalance_ratio(y, minority_label):
        minority_count = int(np.sum(y == minority_label))
        majority_count = int(len(y) - minority_count)
        if minority_count == 0 or majority_count == 0:
            raise ValueError("Both binary groups must contain at least one sample.")
        return majority_count / minority_count

    def _resolved_costs(self, ir):
        direct = {
            "cfo": self.cfo,
            "cfp": self.cfp,
            "cfs": self.cfs,
            "cfb": self.cfb,
        }
        overrides = self.cost_factor or {}
        unknown = set(overrides) - set(_DEFAULT_COSTS)
        if unknown:
            raise ValueError(f"Unknown cost_factor keys: {sorted(unknown)}")

        resolved = {}
        for key, default in _DEFAULT_COSTS.items():
            value = overrides.get(key, direct[key])
            if value is None:
                value = default
            value = float(value)
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{key} must be a finite non-negative value.")
            resolved[key] = value * ir if self.scale_costs_by_ir else value
        return resolved

    def _resolved_tree_params(self):
        def _pick(new, old, default, new_name, old_name):
            if new is not None and old is not None and new != old:
                raise ValueError(
                    f"{new_name} and deprecated {old_name} were both supplied "
                    "with different values."
                )
            return new if new is not None else (old if old is not None else default)

        tau_min = float(_pick(self.tau_min, self.tau_pure, _DEFAULT_TAU_MIN,
                              "tau_min", "tau_pure"))
        tau_maj = float(_pick(self.tau_maj, self.tau_out, _DEFAULT_TAU_MAJ,
                              "tau_maj", "tau_out"))
        max_depth = _pick(self.max_depth, self.tree_max_depth, _DEFAULT_MAX_DEPTH,
                          "max_depth", "tree_max_depth")
        min_leaf = _pick(
            self.min_samples_leaf,
            self.tree_min_samples_leaf,
            _DEFAULT_MIN_SAMPLES_LEAF,
            "min_samples_leaf",
            "tree_min_samples_leaf",
        )

        if not 0 <= tau_maj <= 1 or not 0 <= tau_min <= 1:
            raise ValueError("tau_maj and tau_min must lie in [0, 1].")
        if tau_maj >= tau_min:
            raise ValueError("tau_maj must be smaller than tau_min.")
        if max_depth is not None and int(max_depth) < 1:
            raise ValueError("max_depth must be >= 1 or None.")
        if int(min_leaf) < 1:
            raise ValueError("min_samples_leaf must be >= 1.")

        return tau_min, tau_maj, (None if max_depth is None else int(max_depth)), int(min_leaf)

    # ------------------------------------------------------------------
    # Binary weighting engines
    # ------------------------------------------------------------------
    def _binary_weight_details(self, X, y, minority_label=None):
        Xc = self._complexity_X(X)
        y = self._as_1d(y)
        if self._n_samples(Xc) != len(y):
            raise ValueError("X and y have inconsistent numbers of samples.")

        method = self._normalize_method()
        minority_label = self._resolve_minority_label(y, minority_label)
        ir = self._calculate_imbalance_ratio(y, minority_label)
        costs = self._resolved_costs(ir)

        weights = np.ones(len(y), dtype=float)
        minority_idx = np.flatnonzero(y == minority_label)

        details = {
            "minority_label": minority_label,
            "imbalance_ratio": ir,
            "categories": {},
            "category_by_index": {},
            "ambiguity_by_index": {},
            "cluster_ratio_by_index": {},
            "crossfit_folds": None,
        }

        if method == "ncs":
            return weights, details

        if method == "cs":
            mult = float(self.cs_multiplier)
            if not np.isfinite(mult) or mult < 0:
                raise ValueError("cs_multiplier must be a finite non-negative value.")
            weights[minority_idx] = mult * ir
            return weights, details

        if method == "neighbor":
            return self._neighbor_weights(Xc, y, minority_idx, minority_label, costs, details)

        if method == "tree":
            return self._gini_crossfit_weights(
                Xc, y, minority_idx, minority_label, costs, details
            )

        if method == "cluster":
            return self._cluster_weights(
                Xc, y, minority_idx, minority_label, costs, details
            )

        raise RuntimeError("Unexpected method encountered.")

    def _neighbor_weights(self, X, y, minority_idx, minority_label, costs, details):
        n = len(y)
        k = int(self.n_neighbors)
        if k < 1:
            raise ValueError("n_neighbors must be at least 1.")
        if n <= k:
            raise ValueError(
                "n_neighbors must be smaller than the number of training samples."
            )

        nn = NearestNeighbors(n_neighbors=k + 1, metric="euclidean")
        nn.fit(X)
        _, rows = nn.kneighbors(X[minority_idx])

        weights = np.ones(n, dtype=float)
        cats = Counter()
        by_index = {}

        for sample_idx, row in zip(minority_idx, rows):
            row = np.asarray(row, dtype=int)
            neighbors = row[row != sample_idx][:k]
            if len(neighbors) != k:
                # Duplicated zero-distance points can occasionally lead to unusual
                # neighbour ordering. Use the first k returned non-self rows.
                neighbors = np.asarray([j for j in row if j != sample_idx], dtype=int)[:k]
            if len(neighbors) != k:
                raise RuntimeError("Unable to obtain k non-self neighbours.")

            majority_count = int(np.count_nonzero(y[neighbors] != minority_label))
            ratio = majority_count / k

            if majority_count == 0:
                cat, key = "pure", "cfp"
            elif ratio < 0.5:
                cat, key = "safe", "cfs"
            elif majority_count < k:
                cat, key = "border", "cfb"
            else:
                cat, key = "outlier", "cfo"

            weights[sample_idx] = costs[key]
            cats[cat] += 1
            by_index[int(sample_idx)] = cat

        details["categories"] = {name: int(cats.get(name, 0)) for name in ("pure", "safe", "border", "outlier")}
        details["category_by_index"] = by_index
        return weights, details


    def _cluster_weights(self, X, y, minority_idx, minority_label, costs, details):
        """Exploratory cluster-based complexity estimator.

        KMeans is fitted separately to the minority and majority groups. For
        each minority observation, complexity is measured by its relative
        distance to the nearest minority and majority cluster centres:

            r_i = d_min / (d_min + d_maj)

        The fixed exploratory thresholds (0.25, 0.50, 0.75) map this score to
        the same four iCost categories used by Neighbor-iCost.
        """
        n_clusters = int(self.n_clusters)
        if n_clusters < 1:
            raise ValueError("n_clusters must be at least 1.")

        thresholds = tuple(float(v) for v in self.cluster_thresholds)
        if len(thresholds) != 3:
            raise ValueError("cluster_thresholds must contain exactly three values.")
        t1, t2, t3 = thresholds
        if not (0.0 <= t1 < t2 < t3 <= 1.0):
            raise ValueError(
                "cluster_thresholds must satisfy 0 <= t1 < t2 < t3 <= 1."
            )

        majority_idx = np.flatnonzero(y != minority_label)
        n_min_clusters = min(n_clusters, len(minority_idx))
        n_maj_clusters = min(n_clusters, len(majority_idx))
        if n_min_clusters < n_clusters or n_maj_clusters < n_clusters:
            warnings.warn(
                f"n_clusters={n_clusters} exceeds the size of at least one binary "
                f"group; class-specific cluster counts were reduced to "
                f"{n_min_clusters} (minority) and {n_maj_clusters} (majority).",
                RuntimeWarning,
                stacklevel=2,
            )

        minority_km = KMeans(
            n_clusters=n_min_clusters,
            n_init=10,
            random_state=self.random_state,
        )
        majority_km = KMeans(
            n_clusters=n_maj_clusters,
            n_init=10,
            random_state=self.random_state,
        )
        minority_km.fit(X[minority_idx])
        majority_km.fit(X[majority_idx])

        X_min = X[minority_idx]
        d_min = pairwise_distances(X_min, minority_km.cluster_centers_).min(axis=1)
        d_maj = pairwise_distances(X_min, majority_km.cluster_centers_).min(axis=1)
        denom = d_min + d_maj
        ratio = np.divide(
            d_min,
            denom,
            out=np.full_like(d_min, 0.5, dtype=float),
            where=denom > 0,
        )
        ratio = np.clip(ratio, 0.0, 1.0)

        weights = np.ones(len(y), dtype=float)
        cats = Counter()
        by_index = {}
        ratio_by_index = {}

        for sample_idx, r in zip(minority_idx, ratio):
            r = float(r)
            if r <= t1:
                cat, key = "pure", "cfp"
            elif r <= t2:
                cat, key = "safe", "cfs"
            elif r < t3:
                cat, key = "border", "cfb"
            else:
                cat, key = "outlier", "cfo"

            weights[sample_idx] = costs[key]
            cats[cat] += 1
            by_index[int(sample_idx)] = cat
            ratio_by_index[int(sample_idx)] = r

        details["categories"] = {name: int(cats.get(name, 0)) for name in ("pure", "safe", "border", "outlier")}
        details["category_by_index"] = by_index
        details["cluster_ratio_by_index"] = ratio_by_index
        return weights, details

    def _gini_crossfit_weights(self, X, y, minority_idx, minority_label, costs, details):
        tau_min, tau_maj, max_depth, min_leaf = self._resolved_tree_params()

        classes, counts = np.unique(y, return_counts=True)
        min_class_count = int(np.min(counts))
        requested = int(self.crossfit_folds)
        if requested < 2:
            raise ValueError("crossfit_folds must be at least 2.")
        if min_class_count < 2:
            raise ValueError(
                "Cross-fitted Gini estimation requires at least two samples in each binary group."
            )
        n_splits = min(requested, min_class_count)
        if n_splits < requested:
            warnings.warn(
                f"crossfit_folds reduced from {requested} to {n_splits} because "
                "the smallest binary group is too small.",
                RuntimeWarning,
                stacklevel=2,
            )

        cv = StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=self.random_state,
        )

        weights = np.ones(len(y), dtype=float)
        cats = Counter()
        by_index = {}
        ambiguity_by_index = {}
        assigned = set()

        for fold, (fit_idx, hold_idx) in enumerate(cv.split(np.zeros(len(y)), y)):
            tree_seed = None if self.random_state is None else int(self.random_state) + 1000 + fold
            tree = DecisionTreeClassifier(
                criterion="gini",
                max_depth=max_depth,
                min_samples_leaf=min_leaf,
                random_state=tree_seed,
            )
            tree.fit(X[fit_idx], y[fit_idx])

            fit_leaf = tree.apply(X[fit_idx])
            hold_leaf = tree.apply(X[hold_idx])

            leaves, inv = np.unique(fit_leaf, return_inverse=True)
            totals = np.bincount(inv)
            minority_counts = np.bincount(
                inv,
                weights=(y[fit_idx] == minority_label).astype(float),
            )
            leaf_pos = {leaf: pos for pos, leaf in enumerate(leaves)}

            for local_idx, global_idx in enumerate(hold_idx):
                if y[global_idx] != minority_label:
                    continue

                leaf = hold_leaf[local_idx]
                pos = leaf_pos.get(leaf)
                if pos is None:
                    # A held-out observation is always routed to a leaf created
                    # from the fit partition, so this is only a defensive fallback.
                    p_min = 0.0
                else:
                    p_min = float(minority_counts[pos] / totals[pos])

                # Normalised binary Gini impurity: 2 * [1 - p^2 - (1-p)^2]
                # = 4 p (1-p), ranging from 0 to 1.
                ambiguity = 4.0 * p_min * (1.0 - p_min)

                if p_min >= tau_min:
                    cat, value = "minority_dominated", costs["cfp"]
                elif p_min <= tau_maj:
                    cat, value = "majority_dominated", costs["cfo"]
                else:
                    cat = "mixed"
                    value = costs["cfs"] + (costs["cfb"] - costs["cfs"]) * ambiguity

                weights[global_idx] = value
                cats[cat] += 1
                by_index[int(global_idx)] = cat
                ambiguity_by_index[int(global_idx)] = float(ambiguity)
                assigned.add(int(global_idx))

        expected = set(map(int, minority_idx))
        if assigned != expected:
            missing = sorted(expected - assigned)
            raise RuntimeError(
                f"Cross-fitted Gini estimation missed {len(missing)} minority observations."
            )

        details["categories"] = {name: int(cats.get(name, 0)) for name in ("minority_dominated", "mixed", "majority_dominated")}
        details["category_by_index"] = by_index
        details["ambiguity_by_index"] = ambiguity_by_index
        details["crossfit_folds"] = n_splits
        return weights, details

    # ------------------------------------------------------------------
    # Public weighting helpers
    # ------------------------------------------------------------------
    def compute_sample_weight(self, X, y):
        """Compute sample weights for a binary problem.

        For multiclass data use :meth:`compute_ovr_sample_weights`, because the
        iCost multiclass extension defines a separate binary weighting problem
        for each one-vs-rest class.
        """
        y_arr = self._as_1d(y)
        if len(np.unique(y_arr)) != 2:
            raise ValueError(
                "compute_sample_weight expects a binary target. For multiclass "
                "targets use compute_ovr_sample_weights()."
            )
        override = self._resolved_minority_override()
        weights, _ = self._binary_weight_details(X, y_arr, override)
        return weights

    def compute_ovr_sample_weights(self, X, y):
        """Return OvR sample-weight vectors for each class in a multiclass target."""
        y_arr = self._as_1d(y)
        classes = np.unique(y_arr)
        if len(classes) < 3:
            raise ValueError("compute_ovr_sample_weights requires at least three classes.")
        if self._resolved_minority_override() is not None:
            raise ValueError(
                "minority_label/positive_label cannot be fixed globally for multiclass OvR; "
                "the less frequent group is determined independently in each subproblem."
            )

        out = {}
        for cls in classes:
            y_bin = (y_arr == cls).astype(int)
            minority = self._resolve_minority_label(y_bin)
            weights, _ = self._binary_weight_details(X, y_bin, minority)
            out[cls] = weights
        return out

    def apply_cost(self, X, y):
        """Backward-compatible alias for :meth:`compute_sample_weight`."""
        return self.compute_sample_weight(X, y)

    def calculate_imbalance_ratio(self, y):
        """Calculate the binary imbalance ratio using the configured minority label."""
        y_arr = self._as_1d(y)
        override = self._resolved_minority_override()
        minority = self._resolve_minority_label(y_arr, override)
        return self._calculate_imbalance_ratio(y_arr, minority)

    # ------------------------------------------------------------------
    # Estimator fitting / prediction
    # ------------------------------------------------------------------
    def _fit_estimator(self, estimator, X, y, weights):
        method = self._normalize_method()
        if not has_fit_parameter(estimator, "sample_weight"):
            if method == "ncs":
                estimator.fit(X, y)
                return estimator
            raise TypeError(
                f"{estimator.__class__.__name__} does not support sample_weight. "
                "Use a classifier that supports sample-weighted fitting."
            )
        estimator.fit(X, y, sample_weight=weights)
        return estimator

    @staticmethod
    def _positive_score(estimator, X):
        if hasattr(estimator, "decision_function"):
            score = np.asarray(estimator.decision_function(X))
            classes = np.asarray(estimator.classes_)
            if score.ndim == 1:
                if len(classes) != 2:
                    raise ValueError("Unexpected decision_function shape.")
                return score if classes[1] == 1 else -score
            return score[:, int(np.flatnonzero(classes == 1)[0])]

        if hasattr(estimator, "predict_proba"):
            proba = np.asarray(estimator.predict_proba(X))
            classes = np.asarray(estimator.classes_)
            return proba[:, int(np.flatnonzero(classes == 1)[0])]

        return np.asarray(estimator.predict(X), dtype=float)

    def fit(self, X, y):
        """Fit iCost on binary or multiclass data."""
        feature_names = None
        if isinstance(X, pd.DataFrame):
            feature_names = np.asarray(X.columns, dtype=object)

        if _sk_validate_data is not None:
            X_checked, y_checked = _sk_validate_data(
                self,
                X,
                y,
                accept_sparse=True,
                dtype=None,
                ensure_2d=True,
                multi_output=False,
                reset=True,
            )
        else:
            X_checked, y_checked = check_X_y(
                X,
                y,
                accept_sparse=True,
                dtype=None,
                ensure_2d=True,
                multi_output=False,
            )
            self.n_features_in_ = int(X_checked.shape[1])
            if feature_names is not None:
                self.feature_names_in_ = feature_names

        check_classification_targets(y_checked)
        y_arr = np.asarray(y_checked)

        self.classes_ = np.unique(y_arr)
        if len(self.classes_) < 2:
            raise ValueError("iCost requires at least two classes in y; got 1 class.")

        self._is_multiclass_ = len(self.classes_) > 2

        if not self._is_multiclass_:
            override = self._resolved_minority_override()
            weights, details = self._binary_weight_details(X_checked, y_arr, override)

            self.minority_label_ = details["minority_label"]
            self.imbalance_ratio_ = details["imbalance_ratio"]
            self.sample_weight_ = weights
            self.complexity_summary_ = details["categories"]
            self.complexity_category_by_index_ = details["category_by_index"]
            self.gini_ambiguity_by_index_ = details["ambiguity_by_index"]
            self.cluster_ratio_by_index_ = details["cluster_ratio_by_index"]
            self.crossfit_folds_ = details["crossfit_folds"]

            self.estimator_ = clone(self.base_classifier)
            self._fit_estimator(self.estimator_, X_checked, y_arr, weights)
            return self

        if self._resolved_minority_override() is not None:
            raise ValueError(
                "minority_label/positive_label is only valid for binary fitting. "
                "For multiclass OvR, the less frequent binary group is identified "
                "independently for each class."
            )

        self.estimators_ = []
        self.sample_weight_by_class_ = {}
        self.minority_group_by_class_ = {}
        self.imbalance_ratio_by_class_ = {}
        self.complexity_summary_by_class_ = {}

        for cls in self.classes_:
            y_bin = (y_arr == cls).astype(int)
            minority = self._resolve_minority_label(y_bin)
            weights, details = self._binary_weight_details(X_checked, y_bin, minority)

            estimator = clone(self.base_classifier)
            self._fit_estimator(estimator, X_checked, y_bin, weights)

            self.estimators_.append(estimator)
            self.sample_weight_by_class_[cls] = weights
            self.minority_group_by_class_[cls] = int(minority)
            self.imbalance_ratio_by_class_[cls] = details["imbalance_ratio"]
            self.complexity_summary_by_class_[cls] = details["categories"]

        return self

    def _validate_predict_X(self, X):
        if _sk_validate_data is not None:
            return _sk_validate_data(
                self,
                X,
                reset=False,
                accept_sparse=True,
                dtype=None,
                ensure_2d=True,
            )

        X_checked = check_array(X, accept_sparse=True, dtype=None, ensure_2d=True)
        if X_checked.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X_checked.shape[1]} features, but iCost was fitted with "
                f"{self.n_features_in_} features."
            )
        return X_checked

    def predict(self, X):
        """Predict class labels."""
        check_is_fitted(self, "classes_")
        X = self._validate_predict_X(X)
        if not self._is_multiclass_:
            check_is_fitted(self, "estimator_")
            return self.estimator_.predict(X)

        scores = self.decision_function(X)
        return self.classes_[np.argmax(scores, axis=1)]

    def _decision_function_available(self):
        if getattr(self, "_is_multiclass_", False):
            return True
        return hasattr(self.base_classifier, "decision_function")

    def _predict_proba_available(self):
        return hasattr(self.base_classifier, "predict_proba")

    @available_if(_decision_function_available)
    def decision_function(self, X):
        """Return decision scores.

        For multiclass OvR fitting, returns one positive-class score per OvR
        classifier, with shape ``(n_samples, n_classes)``.
        """
        check_is_fitted(self, "classes_")
        X = self._validate_predict_X(X)
        if not self._is_multiclass_:
            check_is_fitted(self, "estimator_")
            if hasattr(self.estimator_, "decision_function"):
                return self.estimator_.decision_function(X)
            raise AttributeError(
                f"{self.estimator_.__class__.__name__} does not support decision_function."
            )

        check_is_fitted(self, "estimators_")
        return np.column_stack([self._positive_score(est, X) for est in self.estimators_])

    @available_if(_predict_proba_available)
    def predict_proba(self, X):
        """Predict class probabilities when supported by the base classifier."""
        check_is_fitted(self, "classes_")
        X = self._validate_predict_X(X)
        if not self._is_multiclass_:
            check_is_fitted(self, "estimator_")
            if hasattr(self.estimator_, "predict_proba"):
                return self.estimator_.predict_proba(X)
            raise AttributeError(
                f"{self.estimator_.__class__.__name__} does not support predict_proba."
            )

        check_is_fitted(self, "estimators_")
        if not all(hasattr(est, "predict_proba") for est in self.estimators_):
            raise AttributeError(
                f"{self.base_classifier.__class__.__name__} does not support predict_proba."
            )

        probs = np.column_stack([
            np.asarray(est.predict_proba(X))[:, int(np.flatnonzero(np.asarray(est.classes_) == 1)[0])]
            for est in self.estimators_
        ])
        denom = probs.sum(axis=1, keepdims=True)
        denom[denom == 0] = 1.0
        return probs / denom

    def score(self, X, y):
        """Return mean classification accuracy."""
        return accuracy_score(y, self.predict(X))
