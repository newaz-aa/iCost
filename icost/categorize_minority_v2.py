# -*- coding: utf-8 -*-
"""
Helper utilities for analyzing minority-class instance complexity.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


def _to_numpy(X):
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

    if len(classes) < 2:
        raise ValueError("At least two classes are required.")

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
    """
    Categorize minority-class samples using local neighborhood composition.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Feature matrix.

    y : array-like of shape (n_samples,)
        Target labels.

    minority_label : int, str, or None, default=None
        Label to treat as the minority class. If None, the rarest class is used.

    n_neighbors : int, default=5
        Number of nearest neighbors used for categorization.

    return_dataframe : bool, default=False
        If True, return a pandas DataFrame containing sample index, number of
        majority neighbors, and assigned category.

    show_summary : bool, default=False
        If True, print category counts.
    """
    X_np = _to_numpy(X)
    y = _as_1d(y)

    if X_np.shape[0] != len(y):
        raise ValueError("X and y have inconsistent numbers of samples.")

    n_neighbors = int(n_neighbors)
    if n_neighbors < 1:
        raise ValueError("n_neighbors must be at least 1.")

    minority_label = _resolve_minority_label(y, minority_label)
    minority_indices = np.where(y == minority_label)[0]

    if len(y) < 2:
        raise ValueError("At least two samples are required for neighborhood analysis.")

    k = min(n_neighbors + 1, len(y))

    knn = NearestNeighbors(n_neighbors=k)
    knn.fit(X_np)

    _, all_neighbor_indices = knn.kneighbors(X_np[minority_indices])

    categories = {
        "pure": [],
        "safe": [],
        "border": [],
        "outlier": [],
    }
    majority_counts = []
    records = []

    for row, sample_idx in zip(all_neighbor_indices, minority_indices):
        neighbors = [idx for idx in row if idx != sample_idx][:n_neighbors]
        count = int(np.sum(y[neighbors] != minority_label)) if neighbors else 0
        majority_counts.append(count)

        if count == 0:
            category = "pure"
        elif count in (1, 2):
            category = "safe"
        elif count in (3, 4):
            category = "border"
        else:
            category = "outlier"

        categories[category].append(int(sample_idx))
        records.append(
            {
                "index": int(sample_idx),
                "majority_neighbors": int(count),
                "category": category,
            }
        )

    majority_counts = np.asarray(majority_counts, dtype=int)

    if show_summary:
        print("Category summary (minority samples):")
        for category_name, indices in categories.items():
            print(f"  {category_name}: {len(indices)}")

    if return_dataframe:
        return pd.DataFrame(records)

    return minority_indices, categories, majority_counts
