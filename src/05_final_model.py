import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib

df = pd.read_csv("../data/water_potability.csv")
X = df.drop(columns=["Potability"])
y = df["Potability"]

feature_names = list(X.columns)

imputer_final = KNNImputer(n_neighbors=5)
X_imputed = imputer_final.fit_transform(X)

scaler_final = StandardScaler()
X_scaled = scaler_final.fit_transform(X_imputed)

final_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=20,
    min_samples_leaf=4,
    random_state=42,
    class_weight="balanced"
)
final_model.fit(X_scaled, y)

print("Final model trained on full dataset:", X.shape)


feature_stats = {}
for col in feature_names:
    feature_stats[col] = {
        "min": float(df[col].min()),
        "max": float(df[col].max()),
        "median": float(df[col].median()),
    }

bundle = {
    "model": final_model,
    "imputer": imputer_final,
    "scaler": scaler_final,
    "feature_names": feature_names,
    "feature_stats": feature_stats,
}

joblib.dump(bundle, "../models/final_model_bundle.joblib")

print("Saved bundle to ../models/final_model_bundle.joblib")