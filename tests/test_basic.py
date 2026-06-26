import numpy as np

from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from icost import iCost, categorize_minority_class, __version__


def _data():
    X, y = make_classification(
        n_samples=200,
        n_features=8,
        n_informative=4,
        n_redundant=1,
        weights=[0.85, 0.15],
        random_state=42,
    )
    return train_test_split(X, y, stratify=y, random_state=42)


def test_version():
    assert __version__ == "0.2.0"


def test_neighbor_fit_predict():
    X_train, X_test, y_train, _ = _data()
    model = iCost(
        base_classifier=LogisticRegression(max_iter=1000),
        method="neighbor",
    )
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    assert pred.shape[0] == X_test.shape[0]
    assert hasattr(model, "sample_weight_")
    assert np.all(model.sample_weight_ >= 0)


def test_tree_fit_predict():
    X_train, X_test, y_train, _ = _data()
    model = iCost(
        base_classifier=LogisticRegression(max_iter=1000),
        method="tree",
        random_state=42,
    )
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    assert pred.shape[0] == X_test.shape[0]


def test_gini_alias_fit_predict():
    X_train, X_test, y_train, _ = _data()
    model = iCost(
        base_classifier=LogisticRegression(max_iter=1000),
        method="gini",
        random_state=42,
    )
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    assert pred.shape[0] == X_test.shape[0]


def test_cs_weights():
    X_train, _, y_train, _ = _data()
    model = iCost(
        base_classifier=LogisticRegression(max_iter=1000),
        method="cs",
    )
    weights = model.compute_sample_weight(X_train, y_train)
    assert weights.shape[0] == y_train.shape[0]
    assert np.max(weights) >= 1


def test_categorize_minority_class():
    X_train, _, y_train, _ = _data()
    minority_indices, categories, majority_counts = categorize_minority_class(
        X_train,
        y_train,
        n_neighbors=5,
    )
    assert len(minority_indices) == len(majority_counts)
    assert set(categories.keys()) == {"pure", "safe", "border", "outlier"}
