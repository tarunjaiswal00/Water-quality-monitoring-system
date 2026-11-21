from datetime import datetime
from pathlib import Path
import base64
import json

import joblib
import numpy as np
import pandas as pd
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
MODEL_PATH = PROJECT_ROOT / "models" / "final_model_bundle.joblib"
DATA_PATH = PROJECT_ROOT / "data" / "water_potability.csv"
METRICS_PATH = PROJECT_ROOT / "outputs" / "baseline_results.json"
HISTORY_PATH = PROJECT_ROOT / "outputs" / "screening_history.csv"
LOGO_PATH = APP_DIR / "assets" / "neeranalytics_wordmark.png"
ICON_PATH = APP_DIR / "assets" / "neeranalytics_water_icon.png"
BACKGROUND_PATH = APP_DIR / "assets" / "water_landing_background.jpg"

PAGES = ["Home", "Dashboard", "Screen sample", "History"]
PAGE_QUERY = {"Home": "Home", "Dashboard": "Dashboard", "Screen sample": "Screen%20sample", "History": "History"}
FEATURE_INFO = {
    "ph": ("pH", "", "Acidity or alkalinity of the water (0–14 scale)."),
    "Hardness": ("Hardness", "mg/L", "Dissolved calcium and magnesium."),
    "Solids": ("Solids", "ppm", "Total dissolved solids (TDS)."),
    "Chloramines": ("Chloramines", "ppm", "Disinfectant formed from chlorine and ammonia."),
    "Sulfate": ("Sulfate", "mg/L", "Naturally occurring sulfate content."),
    "Conductivity": ("Conductivity", "µS/cm", "Ability of the water to conduct electricity."),
    "Organic_carbon": ("Organic carbon", "ppm", "Total organic carbon (TOC)."),
    "Trihalomethanes": ("Trihalomethanes", "µg/L", "By-products of chlorine treatment."),
    "Turbidity": ("Turbidity", "NTU", "Cloudiness caused by suspended particles."),
}

st.set_page_config(page_title="NeerAnalytics | Water potability screening", page_icon=str(ICON_PATH), layout="wide")

@st.cache_resource
def load_bundle():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

@st.cache_data
def load_metrics():
    if METRICS_PATH.exists():
        return json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    return {}

def go_to(page):
    st.query_params["page"] = page

def clear_sample():
    for feature in active_features:
        st.session_state.pop(f"sample_{feature}", None)
    st.session_state.pop("sample_name", None)

def fill_example():
    examples=[(351,"Potable example 1"),(1181,"Potable example 2"),(468,"Non-potable example 1"),(2993,"Non-potable example 2")]
    next_index=(int(st.session_state.get("example_index",-1))+1)%len(examples)
    st.session_state["example_index"]=next_index
    row_index,name=examples[next_index]
    example=data.iloc[row_index]
    for feature in active_features:
        value=example[feature]
        st.session_state[f"sample_{feature}"]=None if pd.isna(value) else float(value)
    st.session_state["sample_name"]=name

def load_saved_history():
    if not HISTORY_PATH.exists():
        return []
    try:
        return pd.read_csv(HISTORY_PATH).to_dict(orient="records")[::-1]
    except pd.errors.EmptyDataError:
        return []

def save_history_entry(entry):
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([entry]).to_csv(
        HISTORY_PATH,
        mode="a",
        header=not HISTORY_PATH.exists(),
        index=False,
    )

def clear_history():
    st.session_state.history = []
    st.session_state.pop("latest_result", None)
    if HISTORY_PATH.exists():
        HISTORY_PATH.unlink()

def esc(text):
    return (str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;"))

def nav_html(current_page):
    return "".join(
        f'<a class="nav-link{" active" if name == current_page else ""}" href="?page={PAGE_QUERY[name]}" target="_top">{name}</a>'
        for name in PAGES
    )

def render_header(current_page, logo_data):
    logo = f'<img class="brand-logo" src="data:image/png;base64,{logo_data}" alt="NeerAnalytics">' if logo_data else '<span class="brand-wordmark">neeranalytics</span>'
    st.markdown(f'<header class="site-nav"><a class="brand" href="?page=Home" target="_top">{logo}</a><nav class="nav-links">{nav_html(current_page)}</nav></header><div class="site-nav-spacer" aria-hidden="true"></div>', unsafe_allow_html=True)

def render_intro(label, title, description):
    st.markdown(f'<div class="page-intro"><div class="eyebrow">{esc(label)}</div><h1>{esc(title)}</h1><p>{esc(description)}</p></div>', unsafe_allow_html=True)

def card_open(class_name="panel"):
    st.markdown(f'<section class="{class_name}">', unsafe_allow_html=True)

def card_close():
    st.markdown('</section>', unsafe_allow_html=True)

if "history" not in st.session_state:
    st.session_state.history = load_saved_history()
if "page" not in st.session_state:
    st.session_state.page = "Home"
requested_page = st.query_params.get("page")
if requested_page in PAGES:
    st.session_state.page = requested_page
page = st.session_state.page

if not MODEL_PATH.exists():
    st.error(f"Model bundle not found at {MODEL_PATH}. Keep the app folder beside the existing models folder.")
    st.stop()
try:
    model_bundle = load_bundle()
except Exception as exc:
    st.error(f"Could not load the saved model: {exc}")
    st.info("Install the project dependencies in the Python environment used to run Streamlit.")
    st.stop()
active_features = model_bundle["feature_names"]
logo_data = base64.b64encode(LOGO_PATH.read_bytes()).decode("ascii") if LOGO_PATH.exists() else ""
icon_data = base64.b64encode(ICON_PATH.read_bytes()).decode("ascii") if ICON_PATH.exists() else ""
background_data = base64.b64encode(BACKGROUND_PATH.read_bytes()).decode("ascii") if BACKGROUND_PATH.exists() else ""

GLOBAL_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Figtree:wght@400;500;600;700&family=Sora:wght@400;500;600;700&display=swap');
:root{color-scheme:light}
.stApp{background:#f1f6fd;color:#14233f}
[data-testid="stAppViewContainer"]{background:linear-gradient(180deg,#edf5ff 0%,#f5f8fc 52%,#eef5fc 100%)}
.block-container{max-width:1540px;padding:1.2rem clamp(1rem,3vw,3rem) 3rem!important}
#MainMenu,footer{visibility:hidden}
header[data-testid="stHeader"]{display:none!important}
.stApp,[data-testid="stAppViewContainer"]{font-family:"Figtree","Aptos","Segoe UI",sans-serif}
h1,h2,h3{font-family:"Sora","Aptos Display","Segoe UI",sans-serif;letter-spacing:-.025em}
.site-nav{position:fixed;top:.25rem;left:50%;transform:translateX(-50%);z-index:10000;isolation:isolate;display:flex;align-items:center;justify-content:space-between;gap:.8rem;width:calc(100% - 2rem);max-width:1480px;box-sizing:border-box;padding:.25rem .75rem;margin:0;border:1px solid rgba(255,255,255,.98);border-radius:999px;background:rgba(250,252,255,.985);backdrop-filter:blur(22px);-webkit-backdrop-filter:blur(22px);box-shadow:0 8px 24px rgba(24,60,105,.14)}
div[data-testid="stElementContainer"]:has(.site-nav){position:relative;z-index:10001}
.site-nav,.site-nav *{pointer-events:auto!important}
.site-nav-spacer{height:4.6rem}
.brand{display:flex;align-items:center;min-width:200px;text-decoration:none!important}.brand-logo{display:block;width:200px;height:36px;object-fit:contain;object-position:left center}.brand-wordmark{font-size:1.5rem;font-weight:700;color:#2865ad}
.nav-links{display:flex;align-items:center;justify-content:flex-end;gap:.25rem;flex-wrap:nowrap}.nav-link{display:inline-flex;flex:0 0 auto;align-items:center;justify-content:center;min-height:2.25rem;padding:.35rem .85rem;border-radius:999px;text-decoration:none!important;color:#52647c!important;font-size:.875rem;font-weight:500;white-space:nowrap;cursor:pointer;user-select:none;transition:background .18s,color .18s}.nav-link:hover{background:rgba(51,112,207,.10);color:#1b4f9a!important}.nav-link:focus-visible{outline:3px solid #77a9ed;outline-offset:2px}.nav-link.active{background:#2e62be;color:white!important}
.page-intro{padding:2.2rem 1rem 1.4rem}.eyebrow{color:#2a68bf;font-size:.75rem;font-weight:600;letter-spacing:.2em;text-transform:uppercase}.page-intro h1{margin:.45rem 0 .55rem;color:#101d38;font-size:clamp(2.25rem,3.8vw,3rem);font-weight:600;letter-spacing:-.02em;line-height:1.1}.page-intro p{max-width:850px;margin:0;color:#5e6d84;font-size:clamp(1rem,1.4vw,1.125rem);line-height:1.65}
.panel{background:rgba(255,255,255,.86);border:1px solid rgba(255,255,255,.96);border-radius:26px;box-shadow:0 14px 36px rgba(27,66,116,.10);padding:1.65rem}
.dashboard-panel{box-sizing:border-box;margin:0 0 1.2rem;min-width:0;overflow:hidden}.dashboard-panel h3{margin:0 0 .45rem;color:#14213d;font-size:1.35rem;font-weight:600}.dashboard-panel .metric-label{margin:0 0 1rem}.dashboard-panel .table-scroll{max-width:100%;overflow-x:auto}.dashboard-panel .note{margin:1rem -1.65rem -1.65rem;border-radius:0 0 25px 25px}.dashboard-panel .hist-labels span{white-space:nowrap}.dashboard-panel .data-table{min-width:680px}.class-balance-panel{margin-top:.25rem}.dashboard-panel.chart-panel{height:100%;min-height:365px}.dashboard-panel-heading{display:flex;align-items:center;justify-content:space-between;gap:1rem;flex-wrap:wrap}.dashboard-panel-heading span{color:#60718a;font-size:.9rem}
.stApp [data-testid="stVerticalBlockBorderWrapper"]{background:rgba(255,255,255,.9);border:1px solid rgba(255,255,255,.98);border-radius:26px;box-shadow:0 14px 36px rgba(27,66,116,.10);padding:1.1rem}
.st-key-screen-form-card,.st-key-screen-result-card,.st-key-history-card{min-width:0}.st-key-screen-form-card [data-testid="stVerticalBlock"]{gap:.55rem}.st-key-screen-form-card [data-testid="stHorizontalBlock"]{gap:1rem}.st-key-screen-form-card .form-hint{min-height:2.2rem}.st-key-screen-result-card{padding:.2rem}.assessment-result{overflow:hidden;border:1px solid #e1e9f3;border-radius:24px;background:rgba(255,255,255,.94);box-shadow:0 12px 28px rgba(26,63,109,.08)}.assessment-result-head{padding:1.35rem 1.5rem;color:#fff}.assessment-result-head.bad{background:#c45a31}.assessment-result-head.good{background:#328b71}.assessment-result-head .eyebrow{color:rgba(255,255,255,.82);letter-spacing:.16em}.assessment-result-head h2{margin:.35rem 0 .4rem;color:#fff;font-size:1.9rem;font-weight:600;line-height:1.2}.assessment-result-head h2 svg{width:1.35rem;height:1.35rem;vertical-align:-.08em;fill:none;stroke:currentColor;stroke-width:1.8;stroke-linecap:round;stroke-linejoin:round}.assessment-result-head p{margin:0;color:#fff}.assessment-result-body{padding:1.25rem 1.5rem 1.5rem}.probability-heading{display:flex;justify-content:space-between;gap:1rem;align-items:center;color:#5c6c83}.probability-heading strong{color:#14213d;font-size:1.8rem}.probability-track{height:12px;margin:.7rem 0;border-radius:99px;overflow:hidden;background:#e7eef8}.probability-fill{height:100%;border-radius:99px}.probability-caption{margin:0;color:#64748b;font-size:.84rem}.result-warning{margin:1rem 0;padding:1rem;border-radius:18px;background:#fff0df;color:#684b38;line-height:1.5}.result-history-link{display:flex;justify-content:center;margin-top:1rem;padding:.7rem 1rem;border:1px solid #d4deea;border-radius:999px;color:#172541!important;text-decoration:none!important;font-weight:600;background:#f8fbff;box-shadow:0 3px 8px rgba(26,63,109,.08)}.history-count{color:#536680}.history-scroll{width:100%;overflow-x:auto;border-radius:12px}.history-table{min-width:1100px;width:100%;border-collapse:collapse;color:#1c2b45;font-size:.9rem}.history-table th{padding:.75rem .65rem;text-align:left;text-transform:uppercase;letter-spacing:.05em;color:#62718a;font-size:.75rem;font-weight:600;border-bottom:1px solid #dbe4f0;white-space:nowrap}.history-table td{padding:.75rem .65rem;border-bottom:1px solid #e8edf5;vertical-align:top;white-space:nowrap}.history-table .history-sub{display:block;margin-top:.2rem;color:#62718a;font-size:.8rem}.history-badge{display:inline-block;padding:.35rem .65rem;border-radius:999px;background:#fff0df;color:#bd542f;font-weight:600}.history-badge.good{background:#e7f5ef;color:#288166}
.st-key-screen-form-card [data-testid="stTextInput"] input,.st-key-screen-form-card [data-testid="stNumberInput"] input{box-sizing:border-box;height:3.6rem;min-height:3.6rem;padding:0 1rem!important;border:1px solid #d3deeb!important;border-radius:15px!important;box-shadow:0 3px 7px rgba(26,63,109,.10)!important;font-size:1rem!important}.st-key-screen-form-card .stButton>button{min-height:3.35rem!important}.st-key-screen-result-card:has(.assessment-placeholder){min-height:calc(100vh - 7rem);box-sizing:border-box}.st-key-screen-result-card .assessment-placeholder{min-height:calc(100vh - 10rem)}.st-key-screen-result-card .assessment-placeholder .icon-bubble{width:4.8rem;height:4.8rem;margin-bottom:1rem;font-size:1.6rem}.st-key-screen-result-card .assessment-placeholder h3{font-size:1.5rem;margin:.2rem 0 .5rem}.st-key-screen-result-card .assessment-placeholder p{font-size:1rem;line-height:1.55}
.metric-card{height:100%;min-height:145px;background:rgba(255,255,255,.9);border:1px solid rgba(255,255,255,.96);border-radius:25px;padding:1.25rem;box-shadow:0 12px 28px rgba(26,63,109,.10)}
.metric-icon{display:flex;align-items:center;width:1.25rem;height:1.25rem;margin:0 0 1rem;color:#2e66c7}.metric-icon svg{display:block;width:100%;height:100%;fill:none;stroke:currentColor;stroke-width:1.8;stroke-linecap:round;stroke-linejoin:round}
.metric-number{font-size:1.875rem;font-weight:600;letter-spacing:-.03em;color:#13213c}.metric-label{color:#5c6c83;font-size:.875rem;margin-top:.15rem}
.feature-card{height:100%;min-height:170px;background:rgba(255,255,255,.9);border:1px solid white;border-radius:24px;padding:1.05rem;box-shadow:0 12px 30px rgba(26,63,109,.10)}
div[data-testid="stHorizontalBlock"]:has(.feature-card){width:calc(100% - 2rem)!important;max-width:72rem!important;margin-left:auto!important;margin-right:auto!important}
.icon-bubble{width:48px;height:48px;border-radius:50%;display:flex;align-items:center;justify-content:center;background:linear-gradient(135deg,#4e99ed,#173c8c);color:white;font-size:1.3rem;margin-bottom:.75rem}
.feature-card h3{color:#15213a;font-size:1.25rem;font-weight:600;margin:0 0 .5rem}.feature-card p{color:#5e6d84;line-height:1.55;margin:0}.feature-card a{display:inline-block;margin-top:1rem;color:#2865be!important;text-decoration:none!important;font-weight:600}
.icon-bubble svg{width:1.4rem;height:1.4rem;fill:none;stroke:currentColor;stroke-width:1.8;stroke-linecap:round;stroke-linejoin:round}
.mission-panel{display:grid;grid-template-columns:1fr 1.15fr;gap:2rem;align-items:center;box-sizing:border-box;width:calc(100% - 2rem);max-width:72rem;margin:0 auto;background:rgba(255,255,255,.82);backdrop-filter:blur(22px);-webkit-backdrop-filter:blur(22px);border:1px solid rgba(255,255,255,.96);border-radius:28px;padding:1.8rem;box-shadow:0 14px 34px rgba(24,60,105,.10)}
.mission-panel h2{font-size:2rem;font-weight:600;letter-spacing:-.02em;color:#12203b;margin:.4rem 0}.mission-panel p{color:#5b6b83;line-height:1.65;font-size:1.05rem}
.mission-confirm{display:flex;align-items:flex-start;gap:.7rem;margin-top:1rem}.mission-confirm svg{width:1.45rem;height:1.45rem;flex:0 0 auto;margin-top:.1rem;fill:none;stroke:#369779;stroke-width:1.8;stroke-linecap:round;stroke-linejoin:round}
.process-step{padding:.25rem .25rem}.process-number{font-size:2.25rem;font-weight:600;color:#cad8f1}.process-step p{color:#5c6c83;line-height:1.45;font-size:1rem}
.process-panel{box-sizing:border-box;width:calc(100% - 3rem);max-width:68rem;margin:1.5rem auto 0;padding:1rem 1.4rem}.process-panel .section-heading{font-size:2rem;margin-bottom:.5rem}.process-grid{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:.8rem}
.home-footer{box-sizing:border-box;width:calc(100% - 2rem);max-width:72rem;margin:2.5rem auto .2rem;background:rgba(255,255,255,.88);backdrop-filter:blur(20px);-webkit-backdrop-filter:blur(20px);border:1px solid rgba(255,255,255,.96);border-radius:28px;padding:1.15rem 1.4rem;color:#52647c;box-shadow:0 14px 34px rgba(24,60,105,.10)}.home-footer-warning{display:flex;align-items:flex-start;gap:.9rem;padding-bottom:1rem;border-bottom:1px solid #d6e1ee;line-height:1.55}.home-footer-warning svg{width:1.45rem;height:1.45rem;flex:0 0 auto;margin-top:.1rem;fill:none;stroke:#cf5c2c;stroke-width:1.8;stroke-linecap:round;stroke-linejoin:round}.home-footer-warning p{margin:0}.home-footer-meta{display:flex;justify-content:space-between;gap:1rem;padding-top:.85rem;font-size:.875rem;color:#536782}
.balance-track{height:15px;border-radius:99px;overflow:hidden;background:#c35a32;display:flex;margin-top:1rem}.balance-potable{background:#3a967d;height:100%}
.bar-list{display:grid;gap:.85rem;margin-top:1rem}.bar-row{display:grid;grid-template-columns:minmax(110px,170px) 1fr 58px;gap:.8rem;align-items:center;color:#52647c;font-size:.9rem}.bar-track{height:12px;border-radius:99px;background:#e7eef8;overflow:hidden}.bar-fill{height:100%;border-radius:99px;background:linear-gradient(90deg,#4b94e3,#2454ae)}
.histogram{display:flex;align-items:flex-end;gap:.35rem;height:200px;margin-top:1rem;border-bottom:1px solid #d5dfec;padding-bottom:.2rem}.hist-bin{flex:1;display:flex;align-items:flex-end;justify-content:center;gap:2px;height:100%;position:relative}.hist-bar{width:40%;min-height:2px;border-radius:5px 5px 0 0}.hist-labels{display:flex;gap:.35rem;justify-content:space-between;color:#6a7789;font-size:.72rem;margin-top:.45rem}.legend{display:flex;gap:1.2rem;color:#63728a;font-size:.84rem;margin-top:.8rem}.legend span::before{content:"";display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:.35rem;background:#3a967d}.legend .nonpotable::before{background:#c35a32}
.data-table{width:100%;border-collapse:collapse;font-size:.9rem;color:#20304a}.data-table th{text-align:left;color:#62718a;font-weight:700;letter-spacing:.05em;text-transform:uppercase;font-size:.76rem;padding:.8rem .65rem;border-bottom:1px solid #dbe4f0}.data-table td{padding:.75rem .65rem;border-bottom:1px solid #e8edf5}.data-table tr:last-child td{border-bottom:0}.right{text-align:right}
.note{background:#fff0e1;border-radius:0 0 20px 20px;color:#694d3b;padding:1rem 1.2rem;line-height:1.5}.warning{background:#fff7e8;border:1px solid #f3d7a8;color:#765527;padding:1rem 1.15rem;border-radius:15px;line-height:1.55;margin:1rem 0}
.form-label{display:block;padding-bottom:.35rem;font-weight:700;line-height:1.4;color:#15213a}.form-hint{font-size:.84rem;color:#6b7890;margin:.15rem 0 .55rem;line-height:1.45}.assessment-placeholder{min-height:390px;display:flex;flex-direction:column;align-items:center;justify-content:center;text-align:center}.assessment-placeholder h3{color:#17213a}.assessment-placeholder p{color:#65748a;max-width:360px}
.result-good{border-left:5px solid #2f997d;background:#edf8f4;padding:1.2rem;border-radius:14px}.result-bad{border-left:5px solid #cf6644;background:#fff2ec;padding:1.2rem;border-radius:14px}
input,textarea{color:#15213a!important;-webkit-text-fill-color:#15213a!important;background:#fff!important}
[data-testid="stNumberInput"] input{color:#15213a!important;-webkit-text-fill-color:#15213a!important;background:#fff!important;caret-color:#2563b8!important;border-color:#d4dfec!important}
[data-testid="stTextInput"] input{color:#15213a!important;-webkit-text-fill-color:#15213a!important;background:#fff!important}
[data-testid="stWidgetLabel"] p{color:#17233d!important;font-weight:650!important}
.stButton>button,[data-testid="stDownloadButton"] button{border-radius:999px!important;font-weight:700!important;min-height:2.7rem!important}
[data-testid="stMetric"]{background:#fff!important;border-radius:18px;padding:1rem;border:1px solid #e5ebf4}
[data-testid="stMetricLabel"],[data-testid="stMetricLabel"] p,[data-testid="stMetricValue"]{color:#16233e!important}[data-testid="stMetricValue"]{font-size:1.9rem!important}
@media(max-width:800px){.site-nav{border-radius:20px;align-items:flex-start;flex-direction:column}.site-nav-spacer{height:9rem}.brand{min-width:0}.brand-logo{width:180px;height:34px}.nav-links{width:100%;justify-content:flex-start;overflow-x:auto}.mission-panel{grid-template-columns:1fr}.process-grid{grid-template-columns:repeat(2,minmax(0,1fr))}.process-panel{width:calc(100% - 1rem);max-width:none;padding:1.1rem;margin-top:1rem}.page-intro{padding:1.5rem .3rem 1rem}.panel{padding:1.1rem}.feature-card{min-height:180px}.histogram{height:160px}}
</style>
"""
st.markdown(GLOBAL_CSS, unsafe_allow_html=True)
if page == "Home" and background_data:
    st.markdown(f'<style>[data-testid="stAppViewContainer"]{{background-image:linear-gradient(rgba(239,246,255,.10),rgba(239,246,255,.10)),url("data:image/jpeg;base64,{background_data}");background-size:cover;background-position:70% center;background-attachment:fixed}}</style>', unsafe_allow_html=True)

# Resolve model-facing values once, preserving the existing bundle and all training assets.
feature_names = model_bundle["feature_names"]
data = load_data() if DATA_PATH.exists() else pd.DataFrame()
metrics = load_metrics()

# Keep a single consistent, clickable navigation bar across every page.
render_header(page, logo_data)

if page == "Home":
    samples_count = len(data) if not data.empty else 3276
    measurement_count = len(feature_names)
    best_accuracy = max((float(v.get("accuracy", 0)) for v in metrics.values()), default=.647) * 100
    hero_style = "background:transparent"
    st.markdown(f"""
    <section class="landing-hero" style="{hero_style}">
      <div class="hero-copy"><div class="hero-pill"><svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 2.5S5.5 10 5.5 14.5a6.5 6.5 0 0 0 13 0C18.5 10 12 2.5 12 2.5Z"/></svg> WATER POTABILITY SCREENING</div>
        <h1>Clear, explainable<br><span>water screening.</span></h1>
        <p>Helping communities understand their water test results — and know when it's time to seek laboratory confirmation.</p>
        <div class="hero-actions"><a class="primary-link" href="?page=Screen%20sample">Screen a sample&nbsp; →</a><a class="secondary-link" href="?page=Dashboard">View dashboard</a></div>
        <div class="hero-stats"><div><strong>{samples_count:,}</strong><small>samples</small></div><div><strong>{measurement_count}</strong><small>measurements</small></div><div><strong>{best_accuracy:.0f}%</strong><small>best baseline accuracy</small></div></div>
      </div>
    </section>
    <style>
    .block-container:has(.landing-hero){{max-width:1600px!important;padding-top:.3rem!important}}
    .section-space:has(.section-heading){{max-width:72rem;width:calc(100% - 2rem);margin:2rem auto 1rem}}
    .landing-hero{{min-height:calc(100vh - 105px);background:transparent!important;background-size:cover;background-position:center;border-radius:0;display:flex;align-items:center;padding:clamp(2rem,7vw,7rem);box-shadow:none;overflow:hidden}}
    .hero-copy{{max-width:790px}}
    .hero-pill{{display:inline-flex;align-items:center;gap:.55rem;padding:.55rem 1rem;border-radius:999px;background:rgba(255,255,255,.82);border:1px solid rgba(255,255,255,.96);color:#2d68bc;font-weight:600;letter-spacing:.14em;font-size:.75rem}}.hero-pill svg{{width:1rem;height:1rem;fill:none;stroke:currentColor;stroke-width:1.8;stroke-linecap:round;stroke-linejoin:round}}
    .hero-copy h1{{font-size:clamp(3rem,4.3vw,3.75rem);line-height:1.05;letter-spacing:-.035em;color:#111d39;margin:1.25rem 0 1rem;font-weight:600}}
    .hero-copy h1 span{{background:linear-gradient(100deg,#4b96e7,#183c88);-webkit-background-clip:text;background-clip:text;color:transparent}}.hero-copy>p{{font-size:clamp(1rem,1.3vw,1.125rem);line-height:1.65;color:#536781;max-width:710px}}
    .hero-actions{{display:flex;flex-wrap:wrap;gap:.85rem;margin:1.6rem 0 1.5rem}}.primary-link,.secondary-link{{display:inline-flex;padding:.85rem 1.35rem;border-radius:999px;text-decoration:none!important;font-weight:600;align-items:center;justify-content:center}}
    .primary-link{{background:linear-gradient(100deg,#4c96e6,#183c88);color:white!important;box-shadow:0 8px 18px rgba(29,74,151,.22)}}.secondary-link{{background:rgba(255,255,255,.85);color:#162440!important;border:1px solid white;box-shadow:0 6px 16px rgba(25,61,105,.08)}}
    .hero-stats{{display:flex;width:max-content;max-width:100%;background:rgba(255,255,255,.88);border:1px solid rgba(255,255,255,.98);border-radius:22px;overflow:hidden;box-shadow:0 10px 22px rgba(28,64,111,.10)}}
    .hero-stats div{{min-width:135px;padding:.75rem 1.1rem;border-right:1px solid #dbe5f2}}.hero-stats div:last-child{{border:0}}.hero-stats strong{{display:block;color:#14213d;font-size:1.875rem;font-weight:600}}.hero-stats small{{color:#5e6e84;font-size:.75rem}}
    .section-space{{margin:2.5rem 0}}.section-heading{{color:#12203a;font-size:2.25rem;font-weight:600;margin:0 0 1rem}}.section-kicker{{color:#2b68bb;letter-spacing:.16em;font-size:.75rem;font-weight:600}}
    @media(max-width:800px){{.landing-hero{{min-height:calc(100svh - 120px);padding:1.4rem;background-position:62% center}}.hero-copy h1{{font-size:3rem}}.hero-stats div{{min-width:98px;padding:.65rem}}.hero-stats strong{{font-size:1.5rem}}.mission-panel{{padding:1.2rem}}.home-footer{{padding:1.15rem}}.home-footer-meta{{flex-direction:column}}}}
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<section class="mission-panel"><div><div class="section-kicker">OUR MISSION</div><h2>Screening, not certification.</h2></div><div><p>NeerAnalytics takes measurements from a water test report and runs them through a machine-learning model trained on 3,276 water samples. It returns an assessment of <b>likely potable</b> or <b>likely non-potable</b>, with the model’s probability for that result.</p><div class="mission-confirm"><svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 22s8-4 8-11V5l-8-3-8 3v6c0 7 8 11 8 11Z"/><path d="m9 12 2 2 4-4"/></svg><p>Use it to decide when a sample deserves a closer look — then confirm with an accredited laboratory.</p></div></div></section>',unsafe_allow_html=True)

    st.markdown('<div class="section-space"><div class="section-kicker">EXPLORE NEERANALYTICS</div><h2 class="section-heading">Tools for clearer water decisions</h2></div>', unsafe_allow_html=True)
    feature_cols=st.columns(3)
    feature_cards=[('<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M9 3h6M10 3v6l-5.5 9.6A1.6 1.6 0 0 0 5.9 21h12.2a1.6 1.6 0 0 0 1.4-2.4L14 9V3"/><path d="M8 15h8"/></svg>',"Screen a sample","Enter up to nine measurements and get a model assessment in seconds.","Screen sample"),('<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M4 19V5M4 19h16M8 15v-4M12 15V7M16 15V9M20 15V4"/></svg>',"Dashboard","Explore the dataset and see how baseline models score on held-out data.","Dashboard"),('<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M3 12a9 9 0 1 0 2.64-6.36L3 8"/><path d="M3 3v5h5M12 7v5l3 2"/></svg>',"History","Review this session’s assessments and download them as a CSV file.","History")]
    for col,(icon,title,desc,target) in zip(feature_cols,feature_cards):
        with col:
            st.markdown(f'<div class="feature-card"><div class="icon-bubble">{icon}</div><h3>{title}</h3><p>{desc}</p><a href="?page={PAGE_QUERY[target]}">Open&nbsp; →</a></div>', unsafe_allow_html=True)

    steps=["Enter measurements from a laboratory water report.","Values are arranged in the model’s expected feature order.","Missing values are filled in and measurements are scaled.","A Random Forest model predicts a class and probability.","See the assessment and keep it in your session history."]
    step_cards="".join(f'<div class="process-step"><div class="process-number">{i:02}</div><p>{step}</p></div>' for i,step in enumerate(steps,1))
    st.markdown(f'<section class="panel process-panel"><h2 class="section-heading">How a screening works</h2><div class="process-grid">{step_cards}</div></section>',unsafe_allow_html=True)
    st.markdown('<div class="home-footer"><div class="home-footer-warning"><svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 22s8-4 8-11V5l-8-3-8 3v6c0 7 8 11 8 11Z"/><path d="M12 8v4M12 16h.01"/></svg><p>NeerAnalytics is a screening aid, not a water-safety certification. The model is imperfect. Confirm drinking-water safety with an accredited laboratory and applicable public-health guidance.</p></div><div class="home-footer-meta"><span>© 2026 NeerAnalytics</span><span>Data: Kaggle Water Potability dataset</span></div></div>',unsafe_allow_html=True)

elif page == "Dashboard":
    render_intro("DASHBOARD","Dataset & model overview","A summary of the Water Potability dataset and how baseline models score on the project’s held-out evaluation.")
    if data.empty:
        st.error("The project dataset could not be loaded.")
        st.stop()
    sample_count=len(data)
    potable_count=int((data["Potability"]==1).sum())
    nonpotable_count=int((data["Potability"]==0).sum())
    missing_count=int(data[feature_names].isna().sum().sum())
    best_accuracy=max((float(v.get("accuracy",0)) for v in metrics.values()),default=0)
    metric_cols=st.columns(4)
    summary=[
        ('<svg viewBox="0 0 24 24" aria-hidden="true"><ellipse cx="12" cy="5" rx="8" ry="3"/><path d="M4 5v14c0 1.7 3.6 3 8 3s8-1.3 8-3V5"/><path d="M4 12c0 1.7 3.6 3 8 3s8-1.3 8-3"/></svg>',f"{sample_count:,}","Samples"),
        ('<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M9 3h6M10 3v6l-5.5 9.6A1.6 1.6 0 0 0 5.9 21h12.2a1.6 1.6 0 0 0 1.4-2.4L14 9V3"/><path d="M8 15h8"/></svg>',str(len(feature_names)),"Measurements"),
        ('<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 2.5S5.5 10 5.5 14.5a6.5 6.5 0 0 0 13 0C18.5 10 12 2.5 12 2.5Z"/></svg>',f"{potable_count/sample_count:.1%}","Marked potable"),
        ('<svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="12" cy="12" r="9"/><circle cx="12" cy="12" r="5.5"/><circle cx="12" cy="12" r="2"/></svg>',f"{missing_count:,}","Missing values"),
    ]
    for col,(icon,number,label) in zip(metric_cols,summary):
        with col: st.markdown(f'<div class="metric-card"><div class="metric-icon">{icon}</div><div class="metric-number">{number}</div><div class="metric-label">{label}</div></div>',unsafe_allow_html=True)

    potable_pct=potable_count/sample_count*100
    st.markdown(f'<section class="panel dashboard-panel class-balance-panel"><div class="dashboard-panel-heading"><h3>Class balance</h3><span>{potable_count:,} potable · {nonpotable_count:,} non-potable</span></div><div class="balance-track"><div class="balance-potable" style="width:{potable_pct:.3f}%"></div></div></section>',unsafe_allow_html=True)

    chart_col1,chart_col2=st.columns(2)
    with chart_col1:
        rows=[]
        if metrics:
            for name,values in metrics.items():
                short={"Logistic Regression":"Logistic","Random Forest":"Random Forest","SVM (RBF)":"SVM"}.get(name,name)
                acc=float(values.get("accuracy",0)); f1=float(values.get("f1",0))
                rows.append(f'<div class="bar-row"><span>{esc(short)}</span><div class="bar-track"><div class="bar-fill" style="width:{acc*100:.2f}%"></div></div><b>{acc:.1%}</b></div><div class="bar-row" style="margin-top:-.55rem"><span style="font-size:.78rem">F1 score</span><div class="bar-track"><div class="bar-fill" style="width:{f1*100:.2f}%;background:#73a9e3"></div></div><small>{f1:.1%}</small></div>')
        baseline_html='<section class="panel dashboard-panel chart-panel"><h3>Baseline model scores</h3><p class="metric-label">Saved project baseline results.</p><div class="bar-list">'+"".join(rows)+'</div></section>'
        st.markdown(baseline_html,unsafe_allow_html=True)
    with chart_col2:
        bins=np.arange(0,16,1)
        values=data[["ph","Potability"]].dropna(subset=["ph"])
        values=values[(values["ph"]>=0)&(values["ph"]<=14)].copy()
        values["interval"]=pd.cut(values["ph"],bins=bins,right=False,include_lowest=True)
        grouped=values.groupby(["interval","Potability"],observed=False).size()
        max_bin=max(int(grouped.max()) if len(grouped) else 1,1)
        bars=[]; labels=[]
        for lo in range(14):
            interval=pd.Interval(lo,lo+1,closed="left")
            pv=int(grouped.get((interval,1),0)); nv=int(grouped.get((interval,0),0))
            ph=max(pv/max_bin*100,1) if pv else 0; nh=max(nv/max_bin*100,1) if nv else 0
            bars.append(f'<div class="hist-bin"><div class="hist-bar" style="height:{ph:.1f}%;background:#3a967d"></div><div class="hist-bar" style="height:{nh:.1f}%;background:#c35a32"></div></div>')
            labels.append(f'<span>{lo}–{lo+1}</span>')
        histogram_html='<section class="panel dashboard-panel chart-panel"><h3>pH distribution</h3><p class="metric-label">Samples per pH interval, split by dataset label.</p><div class="histogram">'+"".join(bars)+'</div><div class="hist-labels">'+"".join(labels)+'</div><div class="legend"><span>Potable</span><span class="nonpotable">Non-potable</span></div></section>'
        st.markdown(histogram_html,unsafe_allow_html=True)

    table=['<table class="data-table"><thead><tr><th>Model</th><th class="right">Accuracy</th><th class="right">Precision</th><th class="right">Recall</th><th class="right">F1</th></tr></thead><tbody>']
    for name,values in metrics.items():
        badge = ' &nbsp; <small style="color:#2b68bb">USED IN APP</small>' if "Random Forest" in name else ""
        table.append(f'<tr><td>{esc(name)}{badge}</td><td class="right">{float(values.get("accuracy",0)):.1%}</td><td class="right">{float(values.get("precision",0)):.1%}</td><td class="right">{float(values.get("recall",0)):.1%}</td><td class="right">{float(values.get("f1",0)):.1%}</td></tr>')
    table.append('</tbody></table>')
    metrics_note=f'<div class="note">Accuracy of {min((float(v.get("accuracy",0)) for v in metrics.values()),default=0):.0%}–{best_accuracy:.0%} means the model is limited and should not be treated as a reliable safety decision by itself. The deployed bundle is trained on the full dataset, so these holdout scores do not validate that exact bundle.</div>'
    st.markdown('<section class="panel dashboard-panel table-panel"><h3>Model metrics</h3><p class="metric-label">Precision, recall and F1 refer to the potable class.</p><div class="table-scroll">'+"".join(table)+'</div>'+metrics_note+'</section>',unsafe_allow_html=True)

    range_rows=[]
    for feature in feature_names:
        label,unit,_=FEATURE_INFO.get(feature,(feature.replace("_"," "),"",""))
        series=data[feature]
        valid=data[[feature,"Potability"]].dropna(subset=[feature])
        pmean=valid.loc[valid["Potability"]==1,feature].mean()
        nmean=valid.loc[valid["Potability"]==0,feature].mean()
        missing=int(series.isna().sum())
        def fmt(v): return "—" if pd.isna(v) else f"{v:,.2f}".rstrip("0").rstrip(".")
        range_rows.append(f'<tr><td>{esc(label)} <small>({esc(unit)})</small></td><td class="right">{fmt(series.min())}</td><td class="right">{fmt(series.max())}</td><td class="right">{fmt(series.mean())}</td><td class="right">{fmt(pmean)}</td><td class="right">{fmt(nmean)}</td><td class="right">{missing if missing else "—"}</td></tr>')
    ranges_html='<section class="panel dashboard-panel table-panel"><h3>Measurement ranges</h3><p class="metric-label">Values observed in the dataset — not recommended health limits.</p><div class="table-scroll"><table class="data-table"><thead><tr><th>Measurement</th><th class="right">Min</th><th class="right">Max</th><th class="right">Mean</th><th class="right">Mean · potable</th><th class="right">Mean · non-potable</th><th class="right">Missing</th></tr></thead><tbody>'+"".join(range_rows)+'</tbody></table></div></section>'
    st.markdown(ranges_html,unsafe_allow_html=True)

elif page == "Screen sample":
    render_intro("SCREEN SAMPLE","Assess a water sample","Enter measurements from your laboratory report. Leave a field blank if it wasn’t tested — the model will impute a value.")
    left,right=st.columns([1.35,1],gap="large")
    with left:
        with st.container(border=True,key="screen-form-card"):
            sample_name=st.text_input("Sample name (optional)",key="sample_name",placeholder="e.g. Village well, north side")
            cols=st.columns(2)
            input_values={}
            for index,feature in enumerate(feature_names):
                label,unit,description=FEATURE_INFO.get(feature,(feature.replace("_"," "),"",""))
                stats=model_bundle["feature_stats"][feature]
                low,high=float(stats["min"]),float(stats["max"])
                if high<=low: high=low+1
                key=f"sample_{feature}"
                if key not in st.session_state: st.session_state[key]=None
                with cols[index%2]:
                    st.markdown(f'<div class="form-label">{esc(label)} <span style="float:right;color:#64748b;font-weight:500">{esc(unit)}</span></div>',unsafe_allow_html=True)
                    input_values[feature]=st.number_input(label,min_value=low,max_value=high,value=st.session_state[key],step=max((high-low)/100,0.01),format="%.3f",key=key,label_visibility="collapsed",placeholder=f"{low:,.2f} – {high:,.2f}")
                    st.markdown(f'<div class="form-hint">{esc(description)}</div>',unsafe_allow_html=True)
            btn1,btn2,btn3=st.columns([1.25,1.15,.8])
            with btn1: submitted=st.button("⚗  Assess sample",type="primary",use_container_width=True)
            with btn2: st.button("✦  Fill example",use_container_width=True,on_click=fill_example)
            with btn3: st.button("↺  Clear",use_container_width=True,on_click=clear_sample)
            st.caption("Fill example cycles automatically through four dataset samples: two potable and two non-potable.")
            st.caption("Placeholder ranges show observed dataset values; they are not recommended health limits.")
    with right:
        with st.container(border=True,key="screen-result-card"):
            if "latest_result" not in st.session_state:
                st.markdown('<div class="assessment-placeholder"><div class="icon-bubble">⚗</div><h3>Your assessment appears here</h3><p>Fill in the measurements you have, then select “Assess sample”.</p></div>',unsafe_allow_html=True)
            else:
                result=st.session_state.latest_result
                predicted=int(result["class"])
                potable_probability=float(result.get("potable_probability",result["score"] if predicted==1 else 1-result["score"]))
                nonpotable_probability=float(result.get("nonpotable_probability",1-potable_probability))
                chosen_probability=potable_probability if predicted==1 else nonpotable_probability
                style="good" if predicted==1 else "bad"
                title="Likely potable" if predicted==1 else "Likely non-potable"
                icon='<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 22s8-4 8-11V5l-8-3-8 3v6c0 7 8 11 8 11Z"/><path d="m9 12 2 2 4-4"/></svg>' if predicted==1 else '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M10.3 3.9 1.8 18.2A2 2 0 0 0 3.5 21h17a2 2 0 0 0 1.7-2.8L13.7 3.9a2 2 0 0 0-3.4 0Z"/><path d="M12 9v4m0 4h.01"/></svg>'
                st.markdown(f'<div class="assessment-result"><div class="assessment-result-head {style}"><div class="eyebrow">MODEL ASSESSMENT</div><h2>{icon} {title}</h2><p>{esc(result["name"])}</p></div><div class="assessment-result-body"><div class="probability-heading"><span>Probability for this result</span><strong>{chosen_probability:.0%}</strong></div><div class="probability-track"><div class="probability-fill" style="width:{chosen_probability*100:.2f}%;background:{"#328b71" if predicted==1 else "#c45a31"}"></div></div><p class="probability-caption">Potable {potable_probability:.0%} · Non-potable {nonpotable_probability:.0%}</p><div class="result-warning">⚠ &nbsp; Consider sending this sample to an accredited laboratory and confirm drinking-water safety before relying on the result.</div><a class="result-history-link" href="?page=History" target="_top">View session history</a></div></div>',unsafe_allow_html=True)
                with st.expander("Review entered measurements"):
                    entered=[]
                    for feature,value in result["values"].items():
                        label,unit,_=FEATURE_INFO.get(feature,(feature.replace("_"," "),"",""))
                        entered.append({"Measurement":label,"Value":value,"Unit":unit})
                    st.dataframe(pd.DataFrame(entered),use_container_width=True,hide_index=True)

    if submitted:
        row_data={name:(np.nan if value is None else float(value)) for name,value in input_values.items()}
        row=pd.DataFrame([[row_data[name] for name in feature_names]],columns=feature_names)
        try:
            prepared=model_bundle["imputer"].transform(row)
            prepared=model_bundle["scaler"].transform(prepared)
            classifier=model_bundle["model"]
            probabilities=classifier.predict_proba(prepared)[0]
            predicted=int(classifier.predict(prepared)[0])
            score=float(probabilities[predicted])
            potable_probability=float(probabilities[1])
            nonpotable_probability=float(probabilities[0])
            result={"class":predicted,"score":score,"potable_probability":potable_probability,"nonpotable_probability":nonpotable_probability,"name":sample_name.strip() or f"Sample {len(st.session_state.history)+1}","values":row_data}
            st.session_state.latest_result=result
            history_row={"Date and time":datetime.now().strftime("%Y-%m-%d %H:%M"),"Sample":result["name"],"Assessment":"Potable" if predicted==1 else "Non-potable","Model score":score}
            history_row.update(row_data)
            st.session_state.history.insert(0,history_row)
            try:
                save_history_entry(history_row)
            except Exception as exc:
                st.warning(f"Assessment completed, but it could not be saved to local history: {exc}")
            st.rerun()
        except Exception as exc:
            st.error(f"The model could not assess this sample: {exc}")

elif page == "History":
    render_intro("HISTORY","Assessment history","Tested samples are saved locally in this project folder and remain available when you reopen the app.")
    history=st.session_state.history
    if history:
        hist_df=pd.DataFrame(history)
        potable_total=int((hist_df["Assessment"]=="Potable").sum())
        with st.container(border=True,key="history-card"):
            controls=st.columns([1,1,1,1])
            controls[0].markdown(f'<div class="history-count">{len(hist_df)} assessment{"s" if len(hist_df)!=1 else ""} · {potable_total} likely potable</div>',unsafe_allow_html=True)
            csv=hist_df.to_csv(index=False).encode("utf-8")
            controls[2].download_button("⇩  Download CSV",data=csv,file_name="neeranalytics_screening_history.csv",mime="text/csv",use_container_width=True)
            controls[3].button("▤  Clear",on_click=clear_history,use_container_width=True)
            history_rows=[]
            for record in history:
                assessment=str(record.get("Assessment",""))
                model_score=record.get("Model score",np.nan)
                try: model_score=float(model_score)
                except (TypeError,ValueError): model_score=np.nan
                if assessment=="Potable": pprob=model_score; badge="good"
                else: pprob=1-model_score if not pd.isna(model_score) else np.nan; badge=""
                nprob=1-pprob if not pd.isna(pprob) else np.nan
                raw_date=str(record.get("Date and time",""))
                try: date_label=datetime.strptime(raw_date,"%Y-%m-%d %H:%M").strftime("%d %b %Y, %I:%M %p").lstrip("0")
                except ValueError: date_label=raw_date
                name=esc(record.get("Sample","Sample"))
                assessment_cell=f'<span class="history-badge {badge}">Likely {esc(assessment.lower())}</span><span class="history-sub">{model_score:.0%} probability</span>' if not pd.isna(model_score) else f'<span class="history-badge {badge}">Likely {esc(assessment.lower())}</span>'
                cells=[f'<td>{name}<span class="history-sub">{esc(date_label)}</span></td>',f'<td>{assessment_cell}</td>']
                for feature in feature_names:
                    value=record.get(feature,np.nan)
                    label,_,_=FEATURE_INFO.get(feature,(feature.replace("_"," "),"",""))
                    display="—" if pd.isna(value) else f"{float(value):,.2f}".rstrip("0").rstrip(".")
                    cells.append(f'<td title="{esc(label)}">{display}</td>')
                history_rows.append("<tr>"+"".join(cells)+"</tr>")
            history_headers="<th>Sample</th><th>Assessment</th>"+"".join(f'<th>{esc(FEATURE_INFO.get(feature,(feature.replace("_"," "),"",""))[0])}</th>' for feature in feature_names)
            st.markdown('<div class="history-scroll"><table class="history-table"><thead><tr>'+history_headers+'</tr></thead><tbody>'+"".join(history_rows)+'</tbody></table></div>',unsafe_allow_html=True)
    else:
        with st.container(border=True,key="history-card"):
            st.markdown('<div class="assessment-placeholder"><div class="icon-bubble">⚗</div><h3>No assessments yet</h3><p>Screen a sample and it will show up here.</p><a class="primary-link" href="?page=Screen%20sample" target="_top">Screen a sample</a></div>',unsafe_allow_html=True)

if page != "Home":
    st.markdown('<div class="app-footer">⚠ &nbsp; NeerAnalytics is a screening aid, not a water-safety certification. Confirm drinking-water safety with an accredited laboratory.<br><br>© 2026 NeerAnalytics <span style="float:right">Data: Kaggle Water Potability dataset</span></div>',unsafe_allow_html=True)


