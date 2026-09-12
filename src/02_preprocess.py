import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("../data/water_potability.csv")

X = df.drop(columns=["Potability"])
y = df["Potability"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)

print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)
print("Train potable %:", y_train.mean() * 100)
print("Test potable %:", y_test.mean() * 100)

from sklearn.impute import KNNImputer

imputer = KNNImputer(n_neighbors=5)

X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

X_train_imputed = pd.DataFrame(X_train_imputed, columns=X.columns, index=X_train.index)
X_test_imputed = pd.DataFrame(X_test_imputed, columns=X.columns, index=X_test.index)

print("Missing values in train after imputation:", X_train_imputed.isnull().sum().sum())
print("Missing values in test after imputation:", X_test_imputed.isnull().sum().sum())

from sklearn.preprocessing import StandardScaler
import joblib

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train_imputed)
X_test_scaled = scaler.transform(X_test_imputed)

X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns, index=X_train.index)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns, index=X_test.index)

# Save everything so later scripts don't need to redo this work
X_train_imputed.to_csv("../data/X_train_imputed.csv", index=False)
X_test_imputed.to_csv("../data/X_test_imputed.csv", index=False)
X_train_scaled.to_csv("../data/X_train_scaled.csv", index=False)
X_test_scaled.to_csv("../data/X_test_scaled.csv", index=False)
y_train.to_csv("../data/y_train.csv", index=False)
y_test.to_csv("../data/y_test.csv", index=False)

joblib.dump(imputer, "../models/imputer.joblib")
joblib.dump(scaler, "../models/scaler.joblib")

print("All preprocessed files saved.")