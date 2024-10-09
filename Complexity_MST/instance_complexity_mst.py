# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 01:53:50 2024

@author: asif newaz
"""

import numpy as np
import pandas as pd
import scipy.sparse

class InstanceComplexity:
    def __init__(self, X, y):
        """
        Initializes the InstanceComplexity class with input data and class labels.
        
        Parameters:
        X (pd.DataFrame): A DataFrame containing the points.
        y (pd.Series): A Series containing class labels for the data points.
        """
        self.X = X.values  # Convert DataFrame to numpy array
        self.y = y.values  # Convert Series to numpy array
        self.meta = self.calculate_meta()  # Calculate metadata
        self.dist_matrix = self.calculate_distance_matrix()

    def calculate_meta(self):
        """
        Calculates metadata indicating the type of each attribute (0 for numerical, 1 for categorical).
        
        Returns:
        numpy.array: An array indicating the type of each attribute.
        """
        meta = np.zeros(self.X.shape[1])
        for i in range(self.X.shape[1]):
            if np.issubdtype(self.X[:, i].dtype, np.number):
                meta[i] = 0  # Numerical
            else:
                meta[i] = 1  # Categorical
        return meta

    def calculate_distance_matrix(self):
        """
        Calculates the distance matrix using the HEOM metric for mixed data types.
        
        Returns:
        numpy.array: A distance matrix for all pairs of points.
        """
        dist_matrix = np.zeros((len(self.X), len(self.X)))
        range_max = np.max(self.X, axis=0)
        range_min = np.min(self.X, axis=0)

        for i in range(len(self.X)):
            for j in range(i + 1, len(self.X)):
                dist = 0
                for k in range(len(self.X[0])):
                    # Handle missing values
                    if self.X[i][k] is None or self.X[j][k] is None:
                        dist += 1
                    # Numerical attributes
                    elif self.meta[k] == 0:
                        if range_max[k] - range_min[k] == 0:
                            dist += (abs(self.X[i][k] - self.X[j][k]))**2
                        else:
                            dist += (abs(self.X[i][k] - self.X[j][k]) / (range_max[k] - range_min[k]))**2
                    # Categorical attributes
                    elif self.meta[k] == 1 and self.X[i][k] != self.X[j][k]:
                        dist += 1

                dist_matrix[i][j] = np.sqrt(dist)
                dist_matrix[j][i] = np.sqrt(dist)

        return dist_matrix

    def calculate_n_inter(self):
        """
        Calculates the minimum spanning tree and counts vertices from distinct classes.
        
        Returns:
        list: Vertices (indices) that have edges connecting them from different classes.
        """
        minimum_spanning_tree = scipy.sparse.csgraph.minimum_spanning_tree(csgraph=np.triu(self.dist_matrix, k=1), overwrite=True)
        mst_array = minimum_spanning_tree.toarray().astype(float)
        
        vertix = []
        for i in range(len(mst_array)):
            for j in range(len(mst_array[0])):
                if mst_array[i][j] != 0 and self.y[i] != self.y[j]:
                    vertix.append(i)
                    vertix.append(j)

        return vertix

    def minority_class_instance(self):
        """
        Returns the indices of minority class instances that are connected to the opposite class.
        
        Returns:
        list: Indices of minority class instances.
        """
        vertix = self.calculate_n_inter()

        minority_indices = vertix[1::2]
        return minority_indices
