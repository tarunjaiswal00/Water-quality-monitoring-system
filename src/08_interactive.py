import joblib
import pandas as pd
import numpy as np
import shap

bundle = joblib.load("../models/final_model_bundle.joblib")
model = bundle["model"]
imputer = bundle["imputer"]
scaler = bundle["scaler"]
feature_names = bundle["feature_names"]
feature_stats = bundle["feature_stats"]

explainer = shap.TreeExplainer(model)

print("=" * 60)
print("AquaCheck - Interactive Prediction")
print("=" * 60)
print("Enter water test values below. Press Enter to use the")
print("typical (median) value for any field you're unsure about.")
print()


user_values = {}

for feature in feature_names:
    median_val = feature_stats[feature]["median"]
    user_input = input(f"{feature} (typical: {median_val:.2f}): ").strip()

    if user_input == "":
        user_values[feature] = median_val
    else:
        user_values[feature] = float(user_input)

print("\nYou entered:")
for k, v in user_values.items():
    print(f"  {k}: {v}")


    row = pd.DataFrame([user_values])[feature_names]
row_imputed = imputer.transform(row)
row_scaled = scaler.transform(row_imputed)

proba = model.predict_proba(row_scaled)[0]
predicted_class = int(np.argmax(proba))
label = "POTABLE" if predicted_class == 1 else "NON-POTABLE"

print("\n" + "=" * 60)
print(f"PREDICTION: {label}")
print(f"Confidence -> Potable: {proba[1]*100:.1f}% | Non-Potable: {proba[0]*100:.1f}%")
print("=" * 60)

shap_values = explainer.shap_values(row_scaled)
class_shap = shap_values[0, :, predicted_class]

print("\nTop reasons for this prediction:")
ranked = sorted(zip(feature_names, class_shap, row.iloc[0].values), key=lambda x: -abs(x[1]))
for feature, shap_val, actual_val in ranked[:5]:
    direction = "supports" if shap_val > 0 else "argues against"
    print(f"  {feature:20s} value={actual_val:<10.2f} SHAP={shap_val:+.4f}  ({direction} '{label}')")