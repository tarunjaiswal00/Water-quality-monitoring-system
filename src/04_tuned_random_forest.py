import pandas as pd

X_train = pd.read_csv("../data/X_train_scaled.csv")
X_test = pd.read_csv("../data/X_test_scaled.csv")
y_train = pd.read_csv("../data/y_train.csv").values.ravel()
y_test = pd.read_csv("../data/y_test.csv").values.ravel()

print("Loaded:", X_train.shape, X_test.shape)


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold

param_grid = {
    "n_estimators": [200, 300, 500],
    "max_depth": [None, 10, 15, 20],
    "min_samples_leaf": [1, 2, 4],
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42, class_weight="balanced"),
    param_grid=param_grid,
    scoring="f1",
    cv=cv,
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)

print("\nBest parameters found:", grid_search.best_params_)
print("Best cross-validation F1 score:", grid_search.best_score_)


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

best_rf = grid_search.best_estimator_

preds_tuned = best_rf.predict(X_test)

print("\n--- Tuned Random Forest (Test Set) ---")
print("Accuracy:", accuracy_score(y_test, preds_tuned))
print("Precision:", precision_score(y_test, preds_tuned))
print("Recall:", recall_score(y_test, preds_tuned))
print("F1 Score:", f1_score(y_test, preds_tuned))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, preds_tuned))