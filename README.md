# AquaCheck — Explainable AI for Water Potability Screening

An EPICS project that predicts water potability from chemical test values
and explains *why* each prediction was made, using SHAP (explainable AI).

## The Idea

A community health worker tests a water sample and gets raw numbers back
(pH, turbidity, hardness, etc.) — numbers that mean little without a lab.
AquaCheck turns those numbers into a fast, explainable safety signal:
likely safe, likely unsafe, or needs lab testing — plus the specific
reasons behind that call, not just a black-box score.

## Project Status: Review 1 Complete

The core ML pipeline is built, tested, and honestly evaluated. The
user-facing web app is planned for Review 3 (see Roadmap below).

## Repository Structure
water-quality-app/
├── data/ # dataset + preprocessed train/test splits
├── src/ # pipeline scripts, run in order
│ ├── 01_eda.py # exploratory analysis (class balance, correlation)
│ ├── 02_preprocess.py # leak-free split, KNN imputation, scaling
│ ├── 03_baseline_models.py # Logistic Regression, Random Forest, SVM
│ ├── 04_tuned_random_forest.py # 5-fold CV hyperparameter search
│ ├── 05_final_model.py # final model bundle for deployment
│ ├── 06_shap_test.py # SHAP explainability proof of concept
│ ├── 07_demo.py # live demo on real labeled samples
│ ├── 08_interactive.py # live demo with user-typed input
│ └── 09_report_visuals.py # confusion matrix, ROC curve for reporting
├── models/ # saved model artifacts (.joblib)
├── outputs/ # generated plots and metrics
└── app/ # Streamlit app (in progress, Review 3)

## Dataset

[Kaggle Water Potability dataset](https://www.kaggle.com/datasets/adityakadiwal/water-potability)
— 3,276 water samples, 9 chemical/physical parameters, binary potability label.

## Key Finding: Honest Evaluation

Several published papers report 90%+ accuracy on this exact dataset. We
investigated why our honest, leak-free pipeline gets ~63-66% and found the
likely cause: common evaluation mistakes such as imputing missing values
on the full dataset before splitting train/test, or evaluating on training
data. We deliberately avoided these mistakes — see `05_leakage_diagnostic`
findings in our progress report for the controlled experiment proving this.

## Model Performance (Honest Holdout Test Set)

| Model | Accuracy | F1 Score |
|---|---|---|
| Logistic Regression | 61.0% | 0.00 |
| Random Forest (default) | 64.7% | 0.50 |
| SVM (RBF kernel) | 61.7% | 0.52 |
| Random Forest (tuned) | 62.7% | 0.51 |

## How to Run

```bash
# Set up environment
python -m venv venv
venv\Scripts\Activate.ps1        # Windows PowerShell
pip install pandas numpy scikit-learn matplotlib seaborn shap streamlit plotly joblib

# Run the pipeline (from inside src/)
cd src
python 01_eda.py
python 02_preprocess.py
python 03_baseline_models.py
python 04_tuned_random_forest.py
python 05_final_model.py

# Live demos
python 07_demo.py           # predicts on real labeled samples
python 08_interactive.py    # type in your own water test values
```

## Roadmap

- **Review 2:** Strengthen the model using a richer contamination dataset
- **Review 3:** Full Streamlit web app — sliders, batch CSV upload, SHAP
  visualizations, About page

## Author

Tarun Jaiswal — EPICS Project