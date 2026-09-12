import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.ensemble import RandomForestClassifier

plt.rcParams['figure.dpi'] = 150
sns.set_style("whitegrid")

# --- Load bundle just for feature_names/imputer/scaler (not the model) ---
bundle = joblib.load("../models/final_model_bundle.joblib")
feature_names = bundle["feature_names"]

# --- Load train/test data ---
X_train = pd.read_csv("../data/X_train_scaled.csv")
y_train = pd.read_csv("../data/y_train.csv").values.ravel()
X_test = pd.read_csv("../data/X_test_scaled.csv")
y_test = pd.read_csv("../data/y_test.csv").values.ravel()

# --- Train the HONEST model (train data only, matches our real reported metrics) ---
honest_model = RandomForestClassifier(
    n_estimators=300, max_depth=20, min_samples_leaf=4,
    random_state=42, class_weight="balanced"
)
honest_model.fit(X_train, y_train)

preds = honest_model.predict(X_test)
proba = honest_model.predict_proba(X_test)[:, 1]

# --- Confusion Matrix ---
cm = confusion_matrix(y_test, preds)
fig, ax = plt.subplots(figsize=(5, 4.5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
            xticklabels=["Non-Potable", "Potable"],
            yticklabels=["Non-Potable", "Potable"], ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_title("Confusion Matrix - Tuned Random Forest (Honest Holdout)")
plt.tight_layout()
plt.savefig("../outputs/report_confusion_matrix.png")
plt.close()
print("Saved confusion matrix.")