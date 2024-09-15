# -*- coding: utf-8 -*-
"""
Created on Sun Sep 15 06:39:34 2024

@author: Asif Newaz
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.neighbors import NearestNeighbors

class iCost(BaseEstimator, ClassifierMixin):
    
    def __init__(self, base_classifier, cfb=None, cfs=None, cfp=None, flag=1):
        """
        base_classifier: A classifier instance (e.g., LogisticRegression(), SVC(), etc.)
        cfb: cost for border examples (default based on imbalance ratio)
        cfs: cost for safe examples (default based on imbalance ratio)
        cfp: cost for pure examples (default based on imbalance ratio)
        flag = 0 --> all minority samples = cost_factor = 1 [original svm]
        flag = 1 --> all minority samples = cost_factor = IR [original cs_svm]
        flag = 2 --> cost applied based on categories
        """
        self.base_classifier = base_classifier
        self.cfb = cfb
        self.cfs = cfs
        self.cfp = cfp
        self.flag = flag

    def calculate_imbalance_ratio(self, y):
        """Calculate the imbalance ratio between the majority and minority classes."""
        class_counts = np.bincount(y)
        majority_count = np.max(class_counts)
        minority_count = np.min(class_counts)
        imbalance_ratio = majority_count / minority_count
        return imbalance_ratio

    def set_default_costs(self, imbalance_ratio):
        """Set default cost values based on the imbalance ratio."""
        if self.cfb is None:
            self.cfb = imbalance_ratio * 1.1  
        if self.cfs is None:
            self.cfs = imbalance_ratio * 0.6
        if self.cfp is None:
            self.cfp = imbalance_ratio * 0.2

    def apply_cost(self, X, y):
        # Calculate the imbalance ratio and set default cost values if they were not provided
        imbalance_ratio = self.calculate_imbalance_ratio(y)
        self.set_default_costs(imbalance_ratio)

        # Calculate sample weights based on cost factor
        sample_weights = np.ones(len(y))
        selected_indices = [i for i in range(len(y)) if y.values[i] == 1]
        
        if self.flag == 0:
            sample_weights[selected_indices]= 1
        elif self.flag == 1:
            sample_weights[selected_indices]= imbalance_ratio
        else:
            yy = self.categorize_minority_class(X, y)
            k = 0
            for idx in selected_indices:
                if yy[k] in ['g6', 'g5', 'g4', 'g3']:
                    sample_weights[idx] = self.cfb
                elif yy[k] == 'g2':
                    sample_weights[idx] = self.cfs
                elif yy[k] == 'g1':
                    sample_weights[idx] = self.cfp
                k += 1
       
        return sample_weights
    
    def fit(self, X, y):
        # Apply the custom cost to the samples
        sample_weights = self.apply_cost(X, y)

        # Check if the classifier supports sample_weight
        if 'sample_weight' in self.base_classifier.fit.__code__.co_varnames:
            # Use sample weights if supported
            self.base_classifier.fit(X, y, sample_weight=sample_weights)
        else:
            # Ignore sample weights if not supported
            print(f"Classifier {self.base_classifier.__class__.__name__} does not support sample_weight.")
            self.base_classifier.fit(X, y)

        return self  # Return self for compatibility with scikit-learn pipeline

    def predict(self, X):
        # Delegate the predict method to the base classifier
        return self.base_classifier.predict(X)
    
    def categorize_minority_class(self, X, y, n_neighbors=6):
        X = pd.DataFrame(X)
        ynp = np.array(y)
    
        knn = NearestNeighbors(n_neighbors=n_neighbors)
        knn.fit(X)
    
        minority_indices = np.where(y == 1)[0]
        distances, indices = knn.kneighbors(X.iloc[minority_indices,:])
    
        ysum = np.sum(ynp[indices] == 0, axis=1)
    
        out = []
        for i in ysum:
            if i == 0:
                out.append('g1')
            elif i == 1:
                out.append('g2')
            elif i == 2:
                out.append('g3')
            elif i == 3:
                out.append('g4')
            elif i == 4:
                out.append('g5')
            elif i > 4:
                out.append('g6')
        
        return out

    # These methods are required by scikit-learn's BaseEstimator
    def get_params(self, deep=True):
        return {
            "base_classifier": self.base_classifier,
            "cfb": self.cfb,
            "cfs": self.cfs,
            "cfp": self.cfp,
            "flag": self.flag,
        }

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self
