# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 01:55:04 2024


@author: Asif Newaz


"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin

class iCost_mst(BaseEstimator, ClassifierMixin):

    def __init__(self, base_classifier, index_vec=None, cfl=None, cfp=None, flag=1):
        """
        base_classifier: A classifier instance (e.g., LogisticRegression(), SVC(), etc.)
        cfl: cost for linked examples (default based on imbalance ratio)
        cfp: cost for pure examples (default based on imbalance ratio)
        flag = 0 --> all minority samples = cost_factor = 1 [original svm]
        flag = 1 --> all minority samples = cost_factor = IR [original cs_svm]
        flag = 2 --> cost applied based on categories
        index_vec = the indices of linked minority class examples. can be obtained from InstanceComplexity class
        """
        self.base_classifier = base_classifier
        self.index_vec = index_vec if index_vec is not None else []  # Initialize index_vec
        self.cfl = cfl
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
        if self.cfl is None:
            self.cfl = imbalance_ratio * 1.2
        if self.cfp is None:
            self.cfp = imbalance_ratio * 0.8

    def predict_proba(self, X):
        """Delegate the predict_proba method to the base classifier if it supports it."""
        if hasattr(self.base_classifier, "predict_proba"):
            return self.base_classifier.predict_proba(X)
        else:
            raise AttributeError(f"The base classifier {self.base_classifier.__class__.__name__} does not support predict_proba.")

    def apply_cost(self, X, y):
        # Calculate the imbalance ratio and set default cost values if they were not provided
        imbalance_ratio = self.calculate_imbalance_ratio(y)
        self.set_default_costs(imbalance_ratio)

        # Calculate sample weights based on cost factor
        sample_weights = np.ones(len(y))

        if isinstance(y, np.ndarray):
            y = pd.DataFrame(y)

        selected_indices = [i for i in range(len(y)) if y.values[i] == 1]

        if self.flag == 0:
            sample_weights[selected_indices]= 1
        elif self.flag == 1:
            sample_weights[selected_indices]= imbalance_ratio
        else:     
            yy = self.categorize_minority_class(X, y, self.index_vec)
            k = 0
            for idx in selected_indices:
                if yy[k] == 'g1':
                    sample_weights[idx] = self.cfl
                elif yy[k] == 'g0':
                    sample_weights[idx] = self.cfp
                k += 1

        return sample_weights
    
    def decision_function(self, X):
        #"""Delegate the decision_function method to the base classifier if it supports it."""
        if hasattr(self.base_classifier, "decision_function"):
            return self.base_classifier.decision_function(X)
        else:
            raise AttributeError(f"The base classifier {self.base_classifier.__class__.__name__} does not support decision_function.")


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

    def categorize_minority_class(self, X, y, index_vec):
        # X,y - dataframe

        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        #X = pd.DataFrame(X)
        ynp = np.array(y)

        minority_indices = np.where(y == 1)[0]

        out = []
        for i in minority_indices:
            if i in index_vec:
                out.append('g1')
            else:
                out.append('g0')

        return out

    # These methods are required by scikit-learn's BaseEstimator
    def get_params(self, deep=True):
        return {
            "base_classifier": self.base_classifier,
            "cfl": self.cfl,
            "cfp": self.cfp,
            "flag": self.flag,
            "index_vec": self.index_vec,
        }

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self