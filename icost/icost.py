# -*- coding: utf-8 -*-
"""
Created on Mon Aug 25 12:37:38 2025

@author: asifn
"""

# -*- coding: utf-8 -*-
"""
iCost: Instance-complexity-aware cost-sensitive wrapper for sklearn estimators.
- MST integration is external: pass `index_vec` (precomputed) or a `linked_indexer` callable.
- Neighbor-based modes: 1/2/3 (with optional custom bin costs for mode=3).
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.neighbors import NearestNeighbors

class iCost(BaseEstimator, ClassifierMixin):
    """
    Parameters
    ----------
    base_classifier : sklearn estimator
        Any sklearn classifier (e.g., LogisticRegression, SVC, etc.).
    method : {'ncs','org','mst','neighbor'}, default='org'
        - 'ncs'     : no cost (all weights = 1)
        - 'org'     : standard CS; minority weighted by imbalance ratio (IR)
        - 'mst'     : use MST-linked indices to split minority into {linked,pure} with costs {cfl,cfp}
        - 'neighbor': neighbor-based bins around minority samples (modes 1/2/3)
    neighbor_mode : {1,2,3} or None
        Required only if method='neighbor'.
    cfs, cfp, cfb, cfo, cfl : float or None
        Costs for Safe / Pure / Border / Outlier / Linked. If None, set relative to IR in set_default_costs().
    index_vec : list[int] or None
        Precomputed linked indices for MST (subset of minority indices).
    linked_indexer : callable(X, y1d, positive_label)->list[int], optional
        Function to compute linked indices on-the-fly for MST.
    neighbor_costs : list[6] or None
        For neighbor_mode=3, custom per-bin costs for opposite-neighbor counts 0..5 (clamped at 5).
    positive_label : int or str, default=1
        Label treated as the positive/minority class for weighting rules.
    """

    def __init__(self, base_classifier,
                 method="org",
                 neighbor_mode=None,
                 cfs=None, cfp=None, cfb=None, cfo=None, cfl=None,
                 index_vec=None,
                 linked_indexer=None,
                 neighbor_costs=None,
                 positive_label=1):
        self.base_classifier = base_classifier
        self.method = method
        self.neighbor_mode = neighbor_mode
        self.cfs = cfs; self.cfp = cfp; self.cfb = cfb; self.cfo = cfo; self.cfl = cfl
        self.index_vec = index_vec if index_vec is not None else []
        self.linked_indexer = linked_indexer
        self.neighbor_costs = neighbor_costs
        self.positive_label = positive_label

    # -------------------- utilities --------------------
    def _as_1d(self, y):
        y = np.asarray(y)
        if y.ndim > 1:
            y = y.ravel()
        return y

    def calculate_imbalance_ratio(self, y1d):
        classes, counts = np.unique(y1d, return_counts=True)
        if self.positive_label in classes:
            pos = counts[classes.tolist().index(self.positive_label)]
            neg = counts.sum() - pos
            minority = max(min(pos, neg), 1)
            majority = max(pos, neg)
        else:
            minority = max(counts.min(), 1)
            majority = counts.max()
        return majority / minority

    def set_default_costs(self, ir):
        # If user didn't supply, set reasonable defaults relative to IR
        if self.cfs is None: self.cfs = ir * 0.6    # safe
        if self.cfp is None: self.cfp = ir * 0.9    # pure
        if self.cfb is None: self.cfb = ir * 1.2    # border
        if self.cfo is None: self.cfo = ir * 0.3    # outlier
        if self.cfl is None: self.cfl = ir * 1.2    # linked

    # -------------------- core weighting --------------------
    def apply_cost(self, X, y):
        y1d = self._as_1d(y)
        ir = self.calculate_imbalance_ratio(y1d)
        self.set_default_costs(ir)

        n = len(y1d)
        w = np.ones(n, dtype=float)
        minority_idx = np.where(y1d == self.positive_label)[0]

        if self.method == "ncs":
            return w

        elif self.method == "org":
            w[minority_idx] = ir
            return w

        elif self.method == "mst":
            # Prefer precomputed indices; else use callable
            if self.index_vec:
                linked = set(int(i) for i in self.index_vec)
            elif callable(self.linked_indexer):
                linked = set(self.linked_indexer(X, y1d, self.positive_label))
            else:
                raise ValueError("MST requires either `index_vec` or a `linked_indexer` callable.")
            for idx in minority_idx:
                w[idx] = self.cfl if idx in linked else self.cfp
            return w

        elif self.method == "neighbor":
            return self.apply_neighbor_cost(X, y1d, minority_idx, ir)

        else:
            raise ValueError(f"Unknown method: {self.method}")

    # -------------------- neighbor-based --------------------
    def apply_neighbor_cost(self, X, y1d, minority_idx, ir, n_neighbors=6):
        # Ensure DataFrame for kneighbors()
        Xdf = pd.DataFrame(X) if isinstance(X, np.ndarray) else X
        # Include self, then drop it
        k = n_neighbors + 1
        knn = NearestNeighbors(n_neighbors=k)
        knn.fit(Xdf)
        _, indices = knn.kneighbors(Xdf.iloc[minority_idx, :])
        nbrs = indices[:, 1:]  # drop self (0th col)
        # Count opposite-class neighbors (opposite of positive_label)
        opp_counts = np.sum(y1d[nbrs] != self.positive_label, axis=1)

        w = np.ones(len(y1d), dtype=float)

        if self.neighbor_mode == 1:
            # safe(0), pure(1–2), border(3–5), else fallback=IR
            for c, idx in zip(opp_counts, minority_idx):
                if c == 0: w[idx] = self.cfs
                elif c in (1, 2): w[idx] = self.cfp
                elif c in (3, 4, 5): w[idx] = self.cfb
                else: w[idx] = ir

        elif self.neighbor_mode == 2:
            # safe(0–1), border(2–4), outlier(>=5)
            for c, idx in zip(opp_counts, minority_idx):
                if c in (0, 1): w[idx] = self.cfs
                elif c in (2, 3, 4): w[idx] = self.cfb
                else: w[idx] = self.cfo

        elif self.neighbor_mode == 3:
            # custom bin costs for c=0..5 (clamped at 5)
            if self.neighbor_costs is None or len(self.neighbor_costs) != 6:
                raise ValueError("neighbor_mode=3 requires neighbor_costs=[g0,g1,g2,g3,g4,g5].")
            for c, idx in zip(opp_counts, minority_idx):
                w[idx] = self.neighbor_costs[min(int(c), 5)]
        else:
            raise ValueError("neighbor_mode must be one of {1,2,3}")

        return w

    # -------------------- sklearn API --------------------
    def fit(self, X, y):
        y1d = self._as_1d(y)
        w = self.apply_cost(X, y1d)
        try:
            self.base_classifier.fit(X, y1d, sample_weight=w)
        except TypeError:
            # estimator doesn't accept sample_weight
            self.base_classifier.fit(X, y1d)
        return self

    def predict(self, X):
        return self.base_classifier.predict(X)

    def predict_proba(self, X):
        if hasattr(self.base_classifier, "predict_proba"):
            return self.base_classifier.predict_proba(X)
        raise AttributeError(f"{self.base_classifier.__class__.__name__} does not support predict_proba.")

    def decision_function(self, X):
        if hasattr(self.base_classifier, "decision_function"):
            return self.base_classifier.decision_function(X)
        raise AttributeError(f"{self.base_classifier.__class__.__name__} does not support decision_function.")

    # -------------------- param API --------------------
    def get_params(self, deep=True):
        return {
            "base_classifier": self.base_classifier,
            "method": self.method,
            "neighbor_mode": self.neighbor_mode,
            "cfs": self.cfs, "cfp": self.cfp, "cfb": self.cfb, "cfo": self.cfo, "cfl": self.cfl,
            "index_vec": self.index_vec,
            "linked_indexer": self.linked_indexer,
            "neighbor_costs": self.neighbor_costs,
            "positive_label": self.positive_label,
        }

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        return self
