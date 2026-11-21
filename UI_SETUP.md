# NeerAnalytics UI setup

Run from the project root, where the `app`, `data`, and `models` folders are located:

```powershell
python -m streamlit run app\streamlit_app.py
```

Keep the files in `app/assets/` in place. The interface uses `neeranalytics_wordmark.png`, `neeranalytics_water_icon.png`, and `neeranalytics_water_hero.png`. Home, Dashboard, Screen sample, and History are clickable pages. Screening still uses the existing model bundle, dataset, and preprocessing pipeline; session history can be downloaded as CSV.

The updated UI is delivered as a separate project copy. It does not replace the original project's README, model, dataset, or training scripts.
