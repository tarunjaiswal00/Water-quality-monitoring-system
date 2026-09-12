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
print("AquaCheck - Live Model Demo")
print("=" * 60)
print("Feature order:", feature_names)
print()

df_full = pd.read_csv("../data/water_potability.csv")

real_potable_row = df_full[df_full["Potability"] == 1].dropna().iloc[0]
real_nonpotable_row = df_full[df_full["Potability"] == 0].dropna().iloc[0]

demo_samples = {
    "Real Sample - Labeled Potable in Dataset": real_potable_row[feature_names].to_dict(),
    "Real Sample - Labeled Non-Potable in Dataset": real_nonpotable_row[feature_names].to_dict(),
}

def predict_and_explain(name, values):
    print("-" * 60)
    print(f"SAMPLE: {name}")
    print("-" * 60)

    row = pd.DataFrame([values])[feature_names]
    row_imputed = imputer.transform(row)
    row_scaled = scaler.transform(row_imputed)

    proba = model.predict_proba(row_scaled)[0]
    predicted_class = int(np.argmax(proba))
    label = "POTABLE" if predicted_class == 1 else "NON-POTABLE"

    print(f"Prediction: {label}")
    print(f"Confidence -> Potable: {proba[1]*100:.1f}% | Non-Potable: {proba[0]*100:.1f}%")

    shap_values = explainer.shap_values(row_scaled)
    class_shap = shap_values[0, :, predicted_class]

    print("\nTop reasons for this prediction:")
    ranked = sorted(zip(feature_names, class_shap, row.iloc[0].values), key=lambda x: -abs(x[1]))
    for feature, shap_val, actual_val in ranked[:4]:
        direction = "supports" if shap_val > 0 else "argues against"
        print(f"  {feature:20s} value={actual_val:<10.2f} SHAP={shap_val:+.4f}  ({direction} '{label}')")
    print()

for name, values in demo_samples.items():
    predict_and_explain(name, values)