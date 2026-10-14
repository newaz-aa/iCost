import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from sklearn.base import clone, is_classifier
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

from icost import iCost, categorize_minority_class, __version__


def _binary_data(n=240):
    X, y = make_classification(
        n_samples=n,
        n_features=8,
        n_informative=4,
        n_redundant=1,
        weights=[0.85, 0.15],
        random_state=42,
    )
    return train_test_split(X, y, stratify=y, random_state=42)


def _multiclass_data():
    X, y = make_classification(
        n_samples=360,
        n_features=10,
        n_informative=6,
        n_redundant=1,
        n_classes=3,
        n_clusters_per_class=1,
        weights=[0.65, 0.25, 0.10],
        random_state=42,
    )
    return train_test_split(X, y, stratify=y, random_state=42)


def test_version():
    assert __version__ == "0.3.0"


def test_sklearn_clone():
    model = iCost(LogisticRegression(max_iter=1000), method="neighbor")
    cloned = clone(model)
    assert cloned.method == "neighbor"
    assert cloned.base_classifier.__class__ is LogisticRegression



def test_sklearn_recognizes_icost_as_classifier():
    model = iCost(LogisticRegression(max_iter=1000), method="neighbor")
    assert is_classifier(model)


def test_default_gini_structural_parameters_match_manuscript():
    model = iCost(LogisticRegression(max_iter=1000), method="tree")
    tau_min, tau_maj, max_depth, min_leaf = model._resolved_tree_params()
    assert tau_min == pytest.approx(0.80)
    assert tau_maj == pytest.approx(0.20)
    assert max_depth == 3
    assert min_leaf == 5


def test_probability_only_classifier_works_with_sklearn_roc_auc_and_aupr():
    X_train, _, y_train, _ = _binary_data()
    model = iCost(
        DecisionTreeClassifier(max_depth=3, random_state=42),
        method="neighbor",
        random_state=42,
    )
    assert not hasattr(model, "decision_function")
    assert hasattr(model, "predict_proba")

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    roc = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc")
    aupr = cross_val_score(model, X_train, y_train, cv=cv, scoring="average_precision")
    assert np.all(np.isfinite(roc))
    assert np.all(np.isfinite(aupr))


def test_cluster_icost_binary_fit_and_relative_distance_scores():
    X_train, X_test, y_train, _ = _binary_data()
    model = iCost(
        LogisticRegression(max_iter=1000),
        method="cluster",
        n_clusters=2,
        random_state=42,
    )
    model.fit(X_train, y_train)
    assert len(model.predict(X_test)) == len(X_test)
    assert set(model.complexity_summary_) == {"pure", "safe", "border", "outlier"}
    minority_n = int(np.sum(y_train == model.minority_label_))
    assert sum(model.complexity_summary_.values()) == minority_n
    assert len(model.cluster_ratio_by_index_) == minority_n
    ratios = np.asarray(list(model.cluster_ratio_by_index_.values()))
    assert np.all((ratios >= 0.0) & (ratios <= 1.0))


def test_cluster_icost_multiclass_fit():
    X_train, X_test, y_train, _ = _multiclass_data()
    model = iCost(
        LogisticRegression(max_iter=1000),
        method="cluster",
        n_clusters=2,
        random_state=42,
    )
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    assert pred.shape == (len(X_test),)
    assert set(model.complexity_summary_by_class_) == set(model.classes_)


def test_neighbor_fit_predict_and_default_weights():
    X_train, X_test, y_train, _ = _binary_data()
    model = iCost(
        LogisticRegression(max_iter=1000),
        method="neighbor",
        random_state=42,
    )
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    assert pred.shape[0] == X_test.shape[0]
    assert hasattr(model, "sample_weight_")
    assert np.all(np.isfinite(model.sample_weight_))
    assert model.complexity_summary_


def test_tree_is_cross_fitted_and_complete():
    X_train, X_test, y_train, _ = _binary_data()
    model = iCost(
        LogisticRegression(max_iter=1000),
        method="tree",
        random_state=42,
    )
    model.fit(X_train, y_train)
    assert model.predict(X_test).shape[0] == len(X_test)
    assert model.crossfit_folds_ == 3
    minority_n = int(np.sum(y_train == model.minority_label_))
    assert sum(model.complexity_summary_.values()) == minority_n
    assert len(model.complexity_category_by_index_) == minority_n


def test_gini_alias():
    X_train, X_test, y_train, _ = _binary_data()
    model = iCost(LogisticRegression(max_iter=1000), method="gini", random_state=42)
    model.fit(X_train, y_train)
    assert model.predict(X_test).shape[0] == len(X_test)


def test_legacy_tree_parameter_aliases_match_canonical():
    X_train, _, y_train, _ = _binary_data()
    a = iCost(
        LogisticRegression(max_iter=1000),
        method="tree",
        tau_min=0.8,
        tau_maj=0.2,
        max_depth=2,
        min_samples_leaf=4,
        random_state=7,
    )
    b = iCost(
        LogisticRegression(max_iter=1000),
        method="tree",
        tau_pure=0.8,
        tau_out=0.2,
        tree_max_depth=2,
        tree_min_samples_leaf=4,
        random_state=7,
    )
    wa = a.compute_sample_weight(X_train, y_train)
    wb = b.compute_sample_weight(X_train, y_train)
    assert np.allclose(wa, wb)


def test_default_costs_are_new_recommended_values():
    X_train, _, y_train, _ = _binary_data()
    model = iCost(LogisticRegression(max_iter=1000), method="neighbor")
    ir = model.calculate_imbalance_ratio(y_train)
    resolved = model._resolved_costs(ir)
    assert resolved["cfo"] == pytest.approx(0.10 * ir)
    assert resolved["cfp"] == pytest.approx(0.30 * ir)
    assert resolved["cfs"] == pytest.approx(0.50 * ir)
    assert resolved["cfb"] == pytest.approx(0.70 * ir)


def test_cs_multiplier():
    X_train, _, y_train, _ = _binary_data()
    model = iCost(
        LogisticRegression(max_iter=1000),
        method="cs",
        cs_multiplier=0.5,
    )
    weights = model.compute_sample_weight(X_train, y_train)
    minority = model._resolve_minority_label(np.asarray(y_train))
    ir = model.calculate_imbalance_ratio(y_train)
    assert np.allclose(weights[np.asarray(y_train) == minority], 0.5 * ir)
    assert np.allclose(weights[np.asarray(y_train) != minority], 1.0)


def test_string_labels_binary():
    X_train, X_test, y_train, _ = _binary_data()
    y_str = np.where(y_train == 1, "rare", "common")
    model = iCost(LogisticRegression(max_iter=1000), method="neighbor")
    model.fit(X_train, y_str)
    pred = model.predict(X_test)
    assert set(np.unique(pred)).issubset({"rare", "common"})
    assert model.minority_label_ == "rare"


def test_dataframe_input():
    X_train, X_test, y_train, _ = _binary_data()
    cols = [f"x{i}" for i in range(X_train.shape[1])]
    X_train = pd.DataFrame(X_train, columns=cols)
    X_test = pd.DataFrame(X_test, columns=cols)
    model = iCost(LogisticRegression(max_iter=1000), method="neighbor")
    model.fit(X_train, y_train)
    assert list(model.feature_names_in_) == cols
    assert len(model.predict(X_test)) == len(X_test)


def test_sparse_input_neighbor_and_tree():
    X_train, X_test, y_train, _ = _binary_data()
    X_train = sparse.csr_matrix(X_train)
    X_test = sparse.csr_matrix(X_test)
    for method in ["neighbor", "tree", "cluster"]:
        model = iCost(
            LogisticRegression(max_iter=1000),
            method=method,
            random_state=42,
        )
        model.fit(X_train, y_train)
        assert len(model.predict(X_test)) == X_test.shape[0]


def test_multiclass_direct_ovr_fit_predict_and_proba():
    X_train, X_test, y_train, _ = _multiclass_data()
    model = iCost(
        LogisticRegression(max_iter=1000),
        method="neighbor",
        random_state=42,
    )
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)
    scores = model.decision_function(X_test)

    assert pred.shape == (len(X_test),)
    assert proba.shape == (len(X_test), 3)
    assert scores.shape == (len(X_test), 3)
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert set(model.sample_weight_by_class_) == set(model.classes_)
    assert set(model.minority_group_by_class_.values()).issubset({0, 1})


def test_multiclass_weight_helper():
    X_train, _, y_train, _ = _multiclass_data()
    model = iCost(LogisticRegression(max_iter=1000), method="tree", random_state=42)
    weights = model.compute_ovr_sample_weights(X_train, y_train)
    assert set(weights) == set(np.unique(y_train))
    assert all(len(w) == len(y_train) for w in weights.values())


def test_old_explicit_one_vs_rest_wrapper_still_works():
    X_train, X_test, y_train, _ = _multiclass_data()
    binary_icost = iCost(
        LogisticRegression(max_iter=1000),
        method="neighbor",
        random_state=42,
    )
    model = OneVsRestClassifier(binary_icost)
    model.fit(X_train, y_train)
    assert len(model.predict(X_test)) == len(X_test)


def test_compute_sample_weight_rejects_multiclass_target():
    X_train, _, y_train, _ = _multiclass_data()
    model = iCost(LogisticRegression(max_iter=1000), method="neighbor")
    with pytest.raises(ValueError, match="binary target"):
        model.compute_sample_weight(X_train, y_train)


def test_ncs_can_use_estimator_without_sample_weight():
    X_train, X_test, y_train, _ = _binary_data()
    model = iCost(KNeighborsClassifier(n_neighbors=3), method="ncs")
    model.fit(X_train, y_train)
    assert len(model.predict(X_test)) == len(X_test)


def test_cost_mode_rejects_estimator_without_sample_weight():
    X_train, _, y_train, _ = _binary_data()
    model = iCost(KNeighborsClassifier(n_neighbors=3), method="neighbor")
    with pytest.raises(TypeError, match="sample_weight"):
        model.fit(X_train, y_train)


def test_categorize_minority_class_general_k():
    X_train, _, y_train, _ = _binary_data()
    for k in [3, 5, 7]:
        minority_indices, categories, majority_counts = categorize_minority_class(
            X_train, y_train, n_neighbors=k
        )
        assert len(minority_indices) == len(majority_counts)
        assert set(categories) == {"pure", "safe", "border", "outlier"}
        assert sum(len(v) for v in categories.values()) == len(minority_indices)


def test_conflicting_legacy_and_new_params_raise():
    X_train, _, y_train, _ = _binary_data()
    model = iCost(
        LogisticRegression(max_iter=1000),
        method="tree",
        tau_min=0.75,
        tau_pure=0.80,
    )
    with pytest.raises(ValueError, match="different values"):
        model.compute_sample_weight(X_train, y_train)
