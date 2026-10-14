"""Direct multiclass OvR usage example for iCost 0.3.0."""

from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import matthews_corrcoef
from sklearn.model_selection import train_test_split

from icost import iCost

X, y = make_classification(
    n_samples=1200,
    n_features=16,
    n_informative=10,
    n_classes=3,
    n_clusters_per_class=1,
    weights=[0.70, 0.20, 0.10],
    random_state=42,
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=42
)

model = iCost(
    base_classifier=LogisticRegression(max_iter=1000),
    method="neighbor",
    random_state=42,
)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("MCC:", round(matthews_corrcoef(y_test, y_pred), 4))
print("OvR imbalance ratios:", model.imbalance_ratio_by_class_)
