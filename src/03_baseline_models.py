import pandas as pd

X_train = pd.read_csv("../data/X_train_scaled.csv")
X_test = pd.read_csv("../data/X_test_scaled.csv")
y_train = pd.read_csv("../data/y_train.csv").values.ravel()
y_test = pd.read_csv("../data/y_test.csv").values.ravel()

print("Loaded:", X_train.shape, X_test.shape)


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train, y_train)

preds = lr.predict(X_test)

print("\n--- Logistic Regression ---")
print("Accuracy:", accuracy_score(y_test, preds))
print("Precision:", precision_score(y_test, preds))
print("Recall:", recall_score(y_test, preds))
print("F1 Score:", f1_score(y_test, preds))


from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced")
rf.fit(X_train, y_train)

preds_rf = rf.predict(X_test)

print("\n--- Random Forest ---")
print("Accuracy:", accuracy_score(y_test, preds_rf))
print("Precision:", precision_score(y_test, preds_rf))
print("Recall:", recall_score(y_test, preds_rf))
print("F1 Score:", f1_score(y_test, preds_rf))


from sklearn.svm import SVC

svm = SVC(kernel="rbf", random_state=42, class_weight="balanced")
svm.fit(X_train, y_train)

preds_svm = svm.predict(X_test)

print("\n--- SVM (RBF) ---")
print("Accuracy:", accuracy_score(y_test, preds_svm))
print("Precision:", precision_score(y_test, preds_svm))
print("Recall:", recall_score(y_test, preds_svm))
print("F1 Score:", f1_score(y_test, preds_svm))


import json

results = {
    "Logistic Regression": {
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds),
        "recall": recall_score(y_test, preds),
        "f1": f1_score(y_test, preds),
    },
    "Random Forest": {
        "accuracy": accuracy_score(y_test, preds_rf),
        "precision": precision_score(y_test, preds_rf),
        "recall": recall_score(y_test, preds_rf),
        "f1": f1_score(y_test, preds_rf),
    },
    "SVM (RBF)": {
        "accuracy": accuracy_score(y_test, preds_svm),
        "precision": precision_score(y_test, preds_svm),
        "recall": recall_score(y_test, preds_svm),
        "f1": f1_score(y_test, preds_svm),
    },
}

with open("../outputs/baseline_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\nSaved results to outputs/baseline_results.json")