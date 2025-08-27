# -*- coding: utf-8 -*-
"""
Created on Mon Aug 25 12:38:34 2025

@author: asifn
"""

# -*- coding: utf-8 -*-
"""
MST-based instance complexity utility
Author: Asif Newaz

This module provides:
- mst_instance_complexity: builds a HEOM distance matrix and MST over data
- compute_mst_linked_indices: convenience function returning minority nodes
  connected to an opposite-class node in the MST
"""

import numpy as np
import pandas as pd
import scipy.sparse


class mst_instance_complexity:
    """
    Build HEOM distance matrix for mixed-type data and compute MST-based
    cross-class connections.
    """

    def __init__(self, X, y):
        """
        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix (numerical + categorical allowed).
        y : pd.Series
            Labels.
        """
        self.X = X.values
        self.y = y.values
        self.meta = self._calculate_meta()
        self.dist_matrix = self._calculate_distance_matrix()

    def _calculate_meta(self):
        """Return 0 for numerical, 1 for categorical per attribute."""
        meta = np.zeros(self.X.shape[1])
        for i in range(self.X.shape[1]):
            if np.issubdtype(self.X[:, i].dtype, np.number):
                meta[i] = 0
            else:
                meta[i] = 1
        return meta

    def _calculate_distance_matrix(self):
        """HEOM distance for mixed data types."""
        n, m = self.X.shape
        dist_matrix = np.zeros((n, n))
        range_max = np.nanmax(self.X, axis=0)
        range_min = np.nanmin(self.X, axis=0)

        for i in range(n):
            for j in range(i + 1, n):
                dist = 0
                for k in range(m):
                    xi, xj = self.X[i][k], self.X[j][k]

                    # Handle missing
                    if (xi is None or xj is None or
                        (isinstance(xi, float) and np.isnan(xi)) or
                        (isinstance(xj, float) and np.isnan(xj))):
                        dist += 1
                    # Numerical
                    elif self.meta[k] == 0:
                        denom = range_max[k] - range_min[k]
                        if denom == 0:
                            dist += (abs(xi - xj))**2
                        else:
                            dist += (abs(xi - xj) / denom)**2
                    # Categorical
                    elif self.meta[k] == 1 and xi != xj:
                        dist += 1

                d = np.sqrt(dist)
                dist_matrix[i, j] = d
                dist_matrix[j, i] = d

        return dist_matrix

    def _calculate_n_inter(self):
        """Return all vertices that appear in inter-class MST edges."""
        mst = scipy.sparse.csgraph.minimum_spanning_tree(
            csgraph=np.triu(self.dist_matrix, k=1), overwrite=True
        )
        mst_array = mst.toarray().astype(float)

        vertices = []
        for i in range(len(mst_array)):
            for j in range(len(mst_array[0])):
                if mst_array[i, j] != 0 and self.y[i] != self.y[j]:
                    vertices.append(i)
                    vertices.append(j)
        return vertices

    def linked_vertices(self):
        """Return all vertices (indices) that connect to opposite-class nodes."""
        return np.unique(self._calculate_n_inter())


def compute_mst_linked_indices(X, y, positive_label=1):
    """
    Return indices of minority samples connected to opposite-class nodes in the MST.

    Parameters
    ----------
    X : array-like or pd.DataFrame
    y : array-like or pd.Series
    positive_label : int or str, default=1

    Returns
    -------
    list of int
    """
    Xdf = pd.DataFrame(X) if isinstance(X, np.ndarray) else X
    yser = pd.Series(y)   if not isinstance(y, pd.Series) else y

    ic = mst_instance_complexity(Xdf, yser)
    verts = ic.linked_vertices()
    return [int(v) for v in verts if yser.iloc[int(v)] == positive_label]
