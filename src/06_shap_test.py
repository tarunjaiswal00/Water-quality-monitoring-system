import joblib
import pandas as pd
import shap

bundle = joblib.load("../models/final_model_bundle.joblib")
model = bundle["model"]
imputer = bundle["imputer"]
scaler = bundle["scaler"]
feature_names = bundle["feature_names"]
feature_stats = bundle["feature_stats"]

explainer = shap.TreeExplainer(model)

print("Bundle loaded. Feature names:", feature_names)


sample = pd.DataFrame([{
    "ph": 3.5,
    "Hardness": feature_stats["Hardness"]["median"],
    "Solids": 45000,
    "Chloramines": feature_stats["Chloramines"]["median"],
    "Sulfate": feature_stats["Sulfate"]["median"],
    "Conductivity": feature_stats["Conductivity"]["median"],
    "Organic_carbon": feature_stats["Organic_carbon"]["median"],
    "Trihalomethanes": feature_stats["Trihalomethanes"]["median"],
    "Turbidity": 6.5,
}])[feature_names]

sample_imputed = imputer.transform(sample)
sample_scaled = scaler.transform(sample_imputed)

proba = model.predict_proba(sample_scaled)
print("\nPredicted probabilities [Non-Potable, Potable]:", proba)

shap_values = explainer.shap_values(sample_scaled)
print("\nSHAP values shape:", shap_values.shape)


import numpy as np

predicted_class = int(np.argmax(proba[0]))
print("\nPredicted class:", "Potable" if predicted_class == 1 else "Non-Potable")

class_shap_values = shap_values[0, :, predicted_class]

for feature, value in sorted(zip(feature_names, class_shap_values), key=lambda x: -abs(x[1])):
    direction = "supports this prediction" if value > 0 else "argues against this prediction"
    print(f"{feature:20s} SHAP={value:+.4f}  ({direction})")