# -*- coding: utf-8 -*-
"""
Created on Wed Aug 27 14:48:44 2025

@author: asifn
"""

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from collections import Counter

def categorize_minority_class(
    data: pd.DataFrame,
    minority_label=1,
    scale=True,
    mode=1,
    custom_bins=None,
    feature_cols=None,
    label_col=None,
    show_summary=True
):
    
    """
   Categorize minority-class samples by count of opposite-class neighbors.

   Parameters
   ----------
   data : DataFrame
       Features + label in one frame.
   minority_label : scalar
       The label value considered "minority".
   n_neighbors : int
       # of neighbors to inspect.
   scale : bool
       Standardize features before KNN.
   mode : int
       1 => safe(0), pure(1–2), border(3–5)
       2 => safe(0–1), border(2–4), outlier(=5)
       3 => return raw g1..g6 buckets (0,1,2,3,4,>=5)
   custom_bins : dict or None
       If provided, overrides mode. Example:
       {
         "s": [0],        # safe if 0
         "p": [1,2],      # pure if 1 or 2
         "b": [3,4],      # border if 3,4
         "o": "ge5"       # outlier if =5
       }
   feature_cols : list[str] or None
       If None, all columns except label_col are used.
   label_col : str or None
       If None, last column is label.

   Returns
   -------
   minority_indices : np.ndarray
       Row indices of minority samples in `data`.
   groups : list
       Category label per minority index (same order).
   opp_counts : np.ndarray
       Opposite-class neighbor counts per minority sample (same order).
   """


    # Select columns
    if label_col is None:
        label_col = data.columns[-1]
    if feature_cols is None:
        feature_cols = [c for c in data.columns if c != label_col]

    X = data[feature_cols].to_numpy()
    y = data[label_col].to_numpy()

    # Optional scaling
    if scale:
        X = StandardScaler().fit_transform(X)

    n_samples = X.shape[0]
    if n_samples < 2:
        raise ValueError("Need at least 2 samples to compute neighbors.")

    # Ask for 6 neighbors (self + 5 nearest). If dataset is tiny, cap safely.
    # We need at least 2 (self + 1) to slice off self and still have >=1 neighbor.
    knn_k = min(6, n_samples)
    if knn_k < 2:
        knn_k = 2

    knn = NearestNeighbors(n_neighbors=knn_k)
    knn.fit(X)

    # Minority indices
    minority_indices = np.where(y == minority_label)[0]
    if minority_indices.size == 0:
        # No minority found: return empty results
        return minority_indices, [], np.array([])

    # Query neighbors for minority samples
    dists, idxs = knn.kneighbors(X[minority_indices, :])

    # Drop the self-neighbor (first column). Keep up to 5 neighbors after that.
    neighbor_idxs = idxs[:, 1:1+min(5, knn_k-1)]

    # Count opposite-class neighbors among those 5 (or fewer if dataset small)
    opp_counts = np.sum(y[neighbor_idxs] != minority_label, axis=1)

    # Map counts to categories
    def assign_mode1(c):
        # safe(0), pure(1–2), border(3–5), else => 'ge6' (not expected here)
        if c == 0: return "s"
        if c in (1, 2): return "p"
        if c in (3, 4, 5): return "b"
        return "ge6"

    def assign_mode2(c):
        # safe(0–1), border(2–4), outlier(5)  [since max is 5 neighbors here]
        if c in (0, 1): return "s"
        if c in (2, 3, 4): return "b"
        return "o"  # c == 5

    def assign_mode3(c):
        # g1..g6 by exact count bucket with max 5 neighbors
        if c == 0: return "g1"
        if c == 1: return "g2"
        if c == 2: return "g3"
        if c == 3: return "g4"
        if c == 4: return "g5"
        return "g6"  # c == 5

    def assign_custom(c, bins: dict):
        # bins may include integers list or special token "ge5"
        for k, vals in bins.items():
            if vals == "ge5" and c >= 5:
                return k
            if isinstance(vals, (list, tuple, set)) and c in vals:
                return k
        return "unassigned"

    if custom_bins is not None:
        groups = [assign_custom(c, custom_bins) for c in opp_counts]
    elif mode == 1:
        groups = [assign_mode1(c) for c in opp_counts]
    elif mode == 2:
        groups = [assign_mode2(c) for c in opp_counts]
    elif mode == 3:
        groups = [assign_mode3(c) for c in opp_counts]
    else:
        raise ValueError("mode must be one of {1,2,3}, or provide custom_bins.")

 # --- Summary ---
    if show_summary:
        counts = Counter(groups)
        print("Category summary (minority samples):")
        for k, v in counts.items():
            print(f"  {k}: {v}")

    return minority_indices, groups, opp_counts
