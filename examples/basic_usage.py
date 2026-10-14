"""Basic binary usage example for iCost."""

from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import matthews_corrcoef
from sklearn.model_selection import train_test_split

from icost import iCost

X, y = make_classification(
    n_samples=1000,
    n_features=20,
    n_informative=8,
    n_redundant=2,
    weights=[0.90, 0.10],
    class_sep=1.0,
    random_state=42,
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=42
)

for method in ["cs", "neighbor", "tree", "cluster"]:
    model = iCost(
        base_classifier=LogisticRegression(max_iter=1000),
        method=method,
        random_state=42,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(method, "MCC=", round(matthews_corrcoef(y_test, y_pred), 4))
