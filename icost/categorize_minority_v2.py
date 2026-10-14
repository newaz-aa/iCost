# -*- coding: utf-8 -*-
"""Helper utilities for analysing minority-class neighbourhood complexity."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.neighbors import NearestNeighbors


def _to_matrix(X):
    if sparse.issparse(X):
        return X.tocsr()
    if isinstance(X, (pd.DataFrame, pd.Series)):
        return X.to_numpy()
    return np.asarray(X)


def _as_1d(y):
    y = np.asarray(y)
    if y.ndim > 1:
        y = y.ravel()
    return y


def _resolve_minority_label(y, minority_label=None):
    classes, counts = np.unique(y, return_counts=True)
    if len(classes) != 2:
        raise ValueError("categorize_minority_class requires exactly two classes.")
    if minority_label is not None:
        if minority_label not in classes:
            raise ValueError("minority_label was not found in y.")
        return minority_label
    return classes[np.argmin(counts)]


def categorize_minority_class(
    X,
    y,
    minority_label=None,
    n_neighbors=5,
    return_dataframe=False,
    show_summary=False,
):
    """Categorize minority samples from local neighbourhood composition.

    Categories follow the iCost ratio rules:

    - pure: ``r = 0``
    - safe: ``0 < r < 0.5``
    - border: ``0.5 <= r < 1``
    - outlier: ``r = 1``

    where ``r`` is the fraction of majority-class observations among the
    ``n_neighbors`` nearest neighbours. With the recommended odd ``k=5``, the
    exact ``r=0.5`` boundary cannot occur.
    """
    X_mat = _to_matrix(X)
    y_arr = _as_1d(y)

    if X_mat.shape[0] != len(y_arr):
        raise ValueError("X and y have inconsistent numbers of samples.")

    k = int(n_neighbors)
    if k < 1:
        raise ValueError("n_neighbors must be at least 1.")
    if len(y_arr) <= k:
        raise ValueError("n_neighbors must be smaller than the number of samples.")

    minority_label = _resolve_minority_label(y_arr, minority_label)
    minority_indices = np.flatnonzero(y_arr == minority_label)

    nn = NearestNeighbors(n_neighbors=k + 1, metric="euclidean")
    nn.fit(X_mat)
    _, all_neighbors = nn.kneighbors(X_mat[minority_indices])

    categories = {"pure": [], "safe": [], "border": [], "outlier": []}
    majority_counts = []
    records = []

    for sample_idx, row in zip(minority_indices, all_neighbors):
        row = np.asarray(row, dtype=int)
        neighbors = row[row != sample_idx][:k]
        if len(neighbors) != k:
            neighbors = np.asarray([j for j in row if j != sample_idx], dtype=int)[:k]
        if len(neighbors) != k:
            raise RuntimeError("Unable to obtain k non-self neighbours.")

        count = int(np.count_nonzero(y_arr[neighbors] != minority_label))
        ratio = count / k

        if count == 0:
            category = "pure"
        elif ratio < 0.5:
            category = "safe"
        elif count < k:
            category = "border"
        else:
            category = "outlier"

        categories[category].append(int(sample_idx))
        majority_counts.append(count)
        records.append(
            {
                "index": int(sample_idx),
                "majority_neighbors": count,
                "majority_neighbor_ratio": float(ratio),
                "category": category,
            }
        )

    if show_summary:
        print("Category summary (minority samples):")
        for name, indices in categories.items():
            print(f"  {name}: {len(indices)}")

    if return_dataframe:
        return pd.DataFrame(records)

    return minority_indices, categories, np.asarray(majority_counts, dtype=int)
