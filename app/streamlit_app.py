"""
Streamlit Application: Indian Code-Mixed Hate Speech Detection
==============================================================

Professional demonstration of trained models for multilingual Indian
code-mixed hate speech detection on the unified multilingual dataset.

Evaluated Models:
- XLM-RoBERTa Base (Primary / Selected Transformer)
- TF-IDF + Logistic Regression (Baseline)
- TF-IDF + Linear SVM (Baseline)

Usage
-----
    streamlit run app/streamlit_app.py
"""

import html
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import load_config, get_project_root
from src.models.xlmr_model import XLMRClassifier
from src.models.baselines import TFIDFLogisticRegression, TFIDFLinearSVM
from src.explain.shap_explainer import XLMRShapExplainer
from src.explain.lime_explainer import XLMRLimeExplainer


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Indian Code-Mixed Hate Speech Detection",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# DESIGN SYSTEM — MODERN SLATE / DARK THEME
# ============================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

/* Global Resets & Typography */
html, body, [class*="css"], [class*="st-"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
}

/* Force dark slate background on entire app */
.stApp {
    background-color: #0b0f19 !important;
    color: #f1f5f9 !important;
}

/* Constrain main block width cleanly without awkward blank space */
.main .block-container {
    max-width: 1180px !important;
    padding: 2rem 2.5rem 3rem 2.5rem !important;
    margin: 0 auto !important;
}

/* Header Styling (High Contrast, Bold, Clean) */
.hero-header {
    margin-bottom: 1.75rem;
    padding-bottom: 1.25rem;
    border-bottom: 1px solid #1e293b;
}
.hero-title {
    font-size: 2.1rem !important;
    font-weight: 700 !important;
    color: #ffffff !important;
    letter-spacing: -0.03em !important;
    line-height: 1.2 !important;
    margin: 0 0 0.4rem 0 !important;
}
.hero-subtitle {
    font-size: 0.98rem !important;
    font-weight: 400 !important;
    color: #94a3b8 !important;
    line-height: 1.5 !important;
    margin: 0 !important;
}

/* Sidebar Styling */
[data-testid="stSidebar"] {
    background-color: #0f172a !important;
    border-right: 1px solid #1e293b !important;
}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
    color: #cbd5e1 !important;
}
.sb-brand {
    font-size: 0.95rem;
    font-weight: 700;
    color: #ffffff;
    letter-spacing: -0.01em;
    margin-bottom: 0.75rem;
}
.sb-item {
    margin-bottom: 1.15rem;
}
.sb-item-label {
    font-size: 0.7rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #64748b;
    margin-bottom: 0.25rem;
}
.sb-item-val {
    font-size: 0.88rem;
    font-weight: 500;
    color: #f1f5f9;
    line-height: 1.4;
}
.sb-badge-ready {
    display: inline-flex;
    align-items: center;
    background: rgba(16, 185, 129, 0.12);
    color: #34d399;
    border: 1px solid rgba(16, 185, 129, 0.25);
    border-radius: 4px;
    padding: 0.2rem 0.6rem;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.02em;
}
.sb-badge-error {
    display: inline-flex;
    align-items: center;
    background: rgba(244, 63, 94, 0.12);
    color: #fb7185;
    border: 1px solid rgba(244, 63, 94, 0.25);
    border-radius: 4px;
    padding: 0.2rem 0.6rem;
    font-size: 0.75rem;
    font-weight: 600;
}
.sb-divider {
    border: none;
    border-top: 1px solid #1e293b;
    margin: 1.1rem 0;
}

/* Tabs Styling */
[data-testid="stTabs"] {
    border-bottom: 1px solid #1e293b !important;
    margin-bottom: 1.75rem !important;
}
[data-testid="stTabs"] button {
    font-size: 0.92rem !important;
    font-weight: 500 !important;
    color: #94a3b8 !important;
    padding: 0.6rem 1.2rem !important;
    border: none !important;
    background: transparent !important;
}
[data-testid="stTabs"] button[aria-selected="true"] {
    color: #818cf8 !important;
    font-weight: 600 !important;
    border-bottom: 2px solid #6366f1 !important;
}

/* Card / Panel Containers */
.ui-card {
    background-color: #131c2e;
    border: 1px solid #1e293b;
    border-radius: 8px;
    padding: 1.5rem;
    margin-bottom: 1.5rem;
}
.ui-card-header {
    font-size: 0.82rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: #94a3b8;
    margin-bottom: 1rem;
}

/* Selectbox styling */
[data-testid="stSelectbox"] label {
    font-size: 0.78rem !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.06em !important;
    color: #94a3b8 !important;
}
[data-testid="stSelectbox"] > div > div {
    background-color: #0f172a !important;
    color: #f8fafc !important;
    border: 1px solid #334155 !important;
    border-radius: 6px !important;
}

/* Text Input Area Styling */
[data-testid="stTextArea"] textarea {
    background-color: #0f172a !important;
    color: #f8fafc !important;
    border: 1px solid #334155 !important;
    border-radius: 6px !important;
    font-size: 0.95rem !important;
    line-height: 1.5 !important;
}
[data-testid="stTextArea"] textarea:focus {
    border-color: #6366f1 !important;
    box-shadow: 0 0 0 1px #6366f1 !important;
}

/* Primary Button Styling */
.stButton > button {
    background-color: #4f46e5 !important;
    color: #ffffff !important;
    border: 1px solid #6366f1 !important;
    border-radius: 6px !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    padding: 0.55rem 1.75rem !important;
    width: auto !important;
    transition: all 0.15s ease-in-out !important;
}
.stButton > button:hover {
    background-color: #4338ca !important;
    border-color: #818cf8 !important;
    color: #ffffff !important;
}

/* Prediction Result Pill Badges */
.result-badge-hate {
    display: inline-block;
    background: rgba(244, 63, 94, 0.12);
    color: #fb7185;
    border: 1px solid rgba(244, 63, 94, 0.35);
    border-radius: 6px;
    padding: 0.5rem 1.5rem;
    font-size: 1.4rem;
    font-weight: 700;
    letter-spacing: 0.03em;
}
.result-badge-nothate {
    display: inline-block;
    background: rgba(16, 185, 129, 0.12);
    color: #34d399;
    border: 1px solid rgba(16, 185, 129, 0.35);
    border-radius: 6px;
    padding: 0.5rem 1.5rem;
    font-size: 1.4rem;
    font-weight: 700;
    letter-spacing: 0.03em;
}
.conf-val {
    font-size: 1.75rem;
    font-weight: 700;
    color: #f8fafc;
    line-height: 1.2;
}

/* Probability Progress Bars */
.prob-item {
    margin-bottom: 0.85rem;
}
.prob-header {
    display: flex;
    justify-content: space-between;
    font-size: 0.85rem;
    font-weight: 500;
    color: #cbd5e1;
    margin-bottom: 0.35rem;
}
.prob-track {
    width: 100%;
    height: 8px;
    background-color: #1e293b;
    border-radius: 4px;
    overflow: hidden;
}
.prob-fill-nothate {
    height: 100%;
    background-color: #10b981;
    border-radius: 4px;
}
.prob-fill-hate {
    height: 100%;
    background-color: #f43f5e;
    border-radius: 4px;
}

/* Performance Metric Cards */
.metric-card {
    background-color: #131c2e;
    border: 1px solid #1e293b;
    border-radius: 8px;
    padding: 1.25rem 1.5rem;
    text-align: center;
}
.metric-title {
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    color: #94a3b8;
    margin-bottom: 0.35rem;
}
.metric-number {
    font-size: 2.1rem;
    font-weight: 700;
    color: #818cf8;
    line-height: 1.1;
}

/* Comparison Table Styling */
.custom-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.88rem;
    margin-top: 0.5rem;
}
.custom-table th {
    background-color: #1e293b;
    color: #cbd5e1;
    font-weight: 600;
    padding: 0.75rem 1rem;
    text-align: left;
    border-bottom: 1px solid #334155;
}
.custom-table td {
    padding: 0.75rem 1rem;
    border-bottom: 1px solid #1e293b;
    color: #e2e8f0;
}
.custom-table tr.highlight td {
    background-color: rgba(99, 102, 241, 0.08);
    color: #ffffff;
    font-weight: 600;
}

/* Dataset / Methodology Stat Box */
.stat-box {
    background-color: #131c2e;
    border: 1px solid #1e293b;
    border-radius: 6px;
    padding: 1rem;
    text-align: center;
}
.stat-box-val {
    font-size: 1.4rem;
    font-weight: 700;
    color: #818cf8;
}
.stat-box-lbl {
    font-size: 0.75rem;
    color: #94a3b8;
    margin-top: 0.2rem;
}

.pill-tag {
    display: inline-block;
    background: #1e293b;
    color: #cbd5e1;
    border: 1px solid #334155;
    border-radius: 4px;
    padding: 0.25rem 0.7rem;
    font-size: 0.8rem;
    font-weight: 500;
    margin-right: 0.4rem;
    margin-bottom: 0.4rem;
}

/* Hide streamlit footer & default decorations */
footer { display: none !important; }
#MainMenu { visibility: hidden !important; }
header { visibility: hidden !important; }
</style>
""", unsafe_allow_html=True)


# ============================================================
# CONFIGURATION & PATHS
# ============================================================

cfg = load_config()
project_root = get_project_root()

LABEL_MAP = {0: "Non-Hate", 1: "Hate"}

UNIFIED_XLMR_PATH = project_root / "models" / "xlmr" / "UNIFIED_XLMR_best"
UNIFIED_TFIDF_LR_PATH = project_root / "models" / "baselines" / "UNIFIED_tfidf_lr.joblib"
UNIFIED_TFIDF_SVM_PATH = project_root / "models" / "baselines" / "UNIFIED_tfidf_svm.joblib"

RESULTS_METRICS = project_root / "results" / "metrics"
RESULTS_FIGURES = project_root / "results" / "figures"

AVAILABLE_MODELS = [
    "XLM-RoBERTa Base (Selected Transformer)",
    "TF-IDF + Logistic Regression (Baseline)",
    "TF-IDF + Linear SVM (Baseline)"
]


# ============================================================
# LOAD VERIFIED EXPERIMENT ARTIFACTS
# ============================================================

def _load_json(path: Path) -> dict:
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {}


XLMR_METRICS_PATH = RESULTS_METRICS / "xlmr_unified_test_metrics_verified.json"
TFIDF_LR_METRICS_PATH = RESULTS_METRICS / "tfidf_lr_unified_test_metrics_verified.json"
TFIDF_SVM_METRICS_PATH = RESULTS_METRICS / "tfidf_svm_unified_test_metrics_verified.json"

XLMR_METRICS = _load_json(XLMR_METRICS_PATH)
TFIDF_LR_METRICS = _load_json(TFIDF_LR_METRICS_PATH)
TFIDF_SVM_METRICS = _load_json(TFIDF_SVM_METRICS_PATH)

RESULTS_ERROR_ANALYSIS = project_root / "results" / "error_analysis"
LANG_METRICS_PATH = RESULTS_ERROR_ANALYSIS / "language_metrics.csv"
LANG_CM_FIG = RESULTS_ERROR_ANALYSIS / "language_confusion_matrices.png"

XLMR_CM_FIG = RESULTS_FIGURES / "xlmr_unified_test_confusion_matrix_verified.png"


def _is_unified_test(m: dict) -> bool:
    """Verify a metrics dict is from the verified unified test set (n=6319)."""
    if not m:
        return False
    return m.get("n_samples") == 6319 and m.get("split") == "test"


BASELINES_VERIFIED = (
    bool(TFIDF_LR_METRICS) and _is_unified_test(TFIDF_LR_METRICS)
    and bool(TFIDF_SVM_METRICS) and _is_unified_test(TFIDF_SVM_METRICS)
)


# ============================================================
# MODEL LOADING (CACHED)
# ============================================================

@st.cache_resource(show_spinner=False)
def load_unified_xlmr():
    """Load the UNIFIED_XLMR_best model once and cache it."""
    if not UNIFIED_XLMR_PATH.exists():
        raise FileNotFoundError(f"Model directory not found: {UNIFIED_XLMR_PATH}")
    required_files = ["config.json", "model.safetensors", "tokenizer_config.json", "tokenizer.json"]
    missing = [f for f in required_files if not (UNIFIED_XLMR_PATH / f).exists()]
    if missing:
        raise FileNotFoundError(f"Required model files missing: {', '.join(missing)}")
    xlmr = XLMRClassifier(config=cfg)
    xlmr.load(str(UNIFIED_XLMR_PATH))
    return xlmr


@st.cache_resource(show_spinner=False)
def load_unified_tfidf_lr():
    """Load the trained UNIFIED TF-IDF Logistic Regression pipeline."""
    if not UNIFIED_TFIDF_LR_PATH.exists():
        raise FileNotFoundError(f"Baseline file not found: {UNIFIED_TFIDF_LR_PATH}")
    return TFIDFLogisticRegression.load(str(UNIFIED_TFIDF_LR_PATH), config=cfg)


@st.cache_resource(show_spinner=False)
def load_unified_tfidf_svm():
    """Load the trained UNIFIED TF-IDF Linear SVM pipeline."""
    if not UNIFIED_TFIDF_SVM_PATH.exists():
        raise FileNotFoundError(f"Baseline file not found: {UNIFIED_TFIDF_SVM_PATH}")
    return TFIDFLinearSVM.load(str(UNIFIED_TFIDF_SVM_PATH), config=cfg)


@st.cache_resource(show_spinner=False)
def load_shap_explainer():
    """
    Build a SHAP explainer backed by the already-cached XLM-R classifier.
    Calls load_unified_xlmr() which returns the same cached instance —
    no second model load occurs.  Model weights are NOT modified.
    """
    xlmr = load_unified_xlmr()       # returns cached XLMRClassifier
    exp  = XLMRShapExplainer()
    exp.load_from_classifier(xlmr)   # no disk I/O for model weights
    return exp


def _render_shap_chart(shap_result: dict, top_k: int = 12) -> None:
    """
    Render a dark-themed horizontal SHAP bar chart inside Streamlit.

    Red  bars = positive SHAP → pushes prediction toward Hate.
    Green bars = negative SHAP → pushes prediction toward Non-Hate.

    Indic-script font warnings are suppressed (values remain correct).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tokens = shap_result["tokens"]
    sv_hof = np.array(shap_result["shap_values_hof"])

    if len(tokens) == 0:
        st.warning("SHAP returned no tokens for this text.")
        return

    # Select top-k by absolute SHAP magnitude, preserve left-to-right order
    if len(tokens) > top_k:
        top_idx = np.argsort(np.abs(sv_hof))[::-1][:top_k]
        top_idx = sorted(top_idx)
        tokens  = [tokens[i] for i in top_idx]
        sv_hof  = sv_hof[top_idx]

    # Clean labels: strip masker-appended spaces; replace empty with middot
    tokens_disp = [t.strip() if t.strip() else "\u00b7" for t in tokens]
    colors      = ["#f43f5e" if v > 0 else "#10b981" for v in sv_hof]
    fig_height  = max(2.8, len(tokens_disp) * 0.46)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")          # suppress font fallback notices
        plt.rcParams["font.sans-serif"] = ["Nirmala UI", "Segoe UI", "DejaVu Sans", "sans-serif"]
        plt.rcParams["axes.unicode_minus"] = False
        fig, ax = plt.subplots(figsize=(9, fig_height))
        fig.patch.set_facecolor("#131c2e")
        ax.set_facecolor("#131c2e")

        y_pos = np.arange(len(tokens_disp))
        ax.barh(y_pos, sv_hof, color=colors,
                edgecolor="#1e293b", linewidth=0.6, height=0.65)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(tokens_disp, fontsize=9.5, color="#f1f5f9")
        ax.invert_yaxis()
        ax.axvline(0, color="#475569", linewidth=1.0, linestyle="--", alpha=0.7)
        ax.tick_params(axis="x", colors="#94a3b8", labelsize=8.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#334155")
        ax.spines["bottom"].set_color("#334155")
        ax.set_xlabel("SHAP contribution toward Hate (HOF)",
                      fontsize=9, color="#94a3b8")
        ax.set_title(
            f"Token Attribution  \u2014  P(Hate) = "
            f"{shap_result['predicted_prob_hof']:.3f}",
            fontsize=9.5, color="#cbd5e1", pad=8,
        )
        fig.tight_layout(pad=0.9)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)


@st.cache_resource(show_spinner=False)
def load_lime_explainer():
    """
    Build a LIME explainer backed by the already-cached XLM-R classifier.
    Calls load_unified_xlmr() which returns the same cached instance —
    no second model load occurs. Model weights are NOT modified.
    """
    xlmr = load_unified_xlmr()       # returns cached XLMRClassifier
    exp  = XLMRLimeExplainer()
    exp.load_from_classifier(xlmr)   # no disk I/O for model weights
    return exp


def _render_lime_chart(lime_result: dict, top_k: int = 10) -> None:
    """
    Render a dark-themed horizontal LIME bar chart inside Streamlit.

    Red  bars = positive weight → pushes prediction toward Hate (explained class).
    Green bars = negative weight → pushes prediction toward Non-Hate.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    features = lime_result["features"][:top_k]
    weights  = lime_result["weights"][:top_k]
    target_name = lime_result["explained_class_name"]

    if len(features) == 0:
        st.warning("LIME returned no features for this text.")
        return

    features_disp = features[::-1]
    weights_disp  = weights[::-1]
    colors        = ["#f43f5e" if w > 0 else "#10b981" for w in weights_disp]
    fig_height    = max(2.8, len(features_disp) * 0.46)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        plt.rcParams["font.sans-serif"] = ["Nirmala UI", "Segoe UI", "DejaVu Sans", "sans-serif"]
        plt.rcParams["axes.unicode_minus"] = False
        fig, ax = plt.subplots(figsize=(9, fig_height))
        fig.patch.set_facecolor("#131c2e")
        ax.set_facecolor("#131c2e")

        y_pos = np.arange(len(features_disp))
        ax.barh(y_pos, weights_disp, color=colors,
                edgecolor="#1e293b", linewidth=0.6, height=0.65)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(features_disp, fontsize=9.5, color="#f1f5f9")
        ax.axvline(0, color="#475569", linewidth=1.0, linestyle="--", alpha=0.7)
        ax.tick_params(axis="x", colors="#94a3b8", labelsize=8.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#334155")
        ax.spines["bottom"].set_color("#334155")
        ax.set_xlabel(f"LIME feature weight (local influence toward {target_name})",
                      fontsize=9, color="#94a3b8")
        ax.set_title(
            f"LIME Feature Attribution  —  Explained: {target_name.upper()}  "
            f"[P(Hate) = {lime_result['probability_hate']:.3f}]",
            fontsize=9.5, color="#cbd5e1", pad=8,
        )
        fig.tight_layout(pad=0.9)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)


# Pre-check active model availability
active_model_status = "Ready"
try:
    _ = load_unified_xlmr()
except Exception as exc:
    active_model_status = f"Error: {exc}"


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown("<div class='sb-brand'>System Overview</div>", unsafe_allow_html=True)
    st.markdown("<hr class='sb-divider'>", unsafe_allow_html=True)

    st.markdown(
        "<div class='sb-item'>"
        "<div class='sb-item-label'>Primary Model</div>"
        "<div class='sb-item-val' style='color:#818cf8;font-weight:600;'>XLM-RoBERTa Base</div>"
        "<div style='font-family:monospace;font-size:0.75rem;color:#64748b;margin-top:0.15rem;'>"
        "UNIFIED_XLMR_best</div>"
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        "<div class='sb-item'>"
        "<div class='sb-item-label'>Evaluated Models (3)</div>"
        "<div style='font-size:0.82rem;color:#94a3b8;line-height:1.6;'>"
        "&bull; TF-IDF + Logistic Regression<br>"
        "&bull; TF-IDF + Linear SVM<br>"
        "&bull; <span style='color:#f1f5f9;font-weight:500;'>XLM-RoBERTa Base</span> (Selected)"
        "</div>"
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        "<div class='sb-item'>"
        "<div class='sb-item-label'>Dataset</div>"
        "<div class='sb-item-val'>Unified Multilingual Dataset</div>"
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        "<div class='sb-item'>"
        "<div class='sb-item-label'>Languages</div>"
        "<div class='sb-item-val'>Tamil &middot; Malayalam &middot; Hindi &middot; Telugu</div>"
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        "<div class='sb-item'>"
        "<div class='sb-item-label'>Task</div>"
        "<div class='sb-item-val'>Binary Classification (Hate / Non-Hate)</div>"
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        "<div class='sb-item'>"
        "<div class='sb-item-label'>Model Status</div>",
        unsafe_allow_html=True,
    )
    if "Error" not in active_model_status:
        st.markdown("<div class='sb-badge-ready'>Ready for Inference</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='sb-badge-error'>Failed to Load</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# MAIN HEADER
# ============================================================

st.markdown(
    "<div class='hero-header'>"
    "<h1 class='hero-title'>Indian Code-Mixed Hate Speech Detection</h1>"
    "<p class='hero-subtitle'>"
    "Multilingual Transformer-Based Detection of Hate Speech in Indian Code-Mixed Text"
    "</p>"
    "</div>",
    unsafe_allow_html=True,
)


# ============================================================
# NAVIGATION TABS
# ============================================================

tab_detect, tab_perf, tab_method = st.tabs(
    ["Detection", "Performance", "Methodology"]
)


# ============================================================
# TAB 1 — DETECTION
# ============================================================

with tab_detect:
    st.markdown(
        "<div style='font-size:0.92rem;color:#cbd5e1;margin-bottom:1.25rem;'>"
        "Select a trained model and enter an Indian code-mixed message below to evaluate it in real time."
        "</div>",
        unsafe_allow_html=True,
    )

    selected_model_name = st.selectbox(
        "Select Model for Live Inference",
        options=AVAILABLE_MODELS,
        index=0,
    )

    st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)

    user_input = st.text_area(
        label="Input Text",
        value="",
        height=110,
        placeholder="Enter Hindi, Tamil, Malayalam, or Telugu code-mixed text...",
        label_visibility="collapsed",
    )

    btn_row_col1, btn_row_col2 = st.columns([1, 4])
    with btn_row_col1:
        analyze_clicked = st.button("Analyze Text", type="primary")
    with btn_row_col2:
        st.markdown(
            "<div style='font-size:0.8rem;color:#64748b;padding-top:0.45rem;'>"
            "Supports native scripts and Romanized/transliterated Indian code-mix."
            "</div>",
            unsafe_allow_html=True,
        )

    if analyze_clicked:
        if not user_input.strip():
            st.warning("Please enter some text to analyze.")
        else:
            with st.spinner(f"Running inference with {selected_model_name.split(' (')[0]}..."):
                try:
                    if "XLM-RoBERTa" in selected_model_name:
                        current_model = load_unified_xlmr()
                        preds, probs = current_model.predict([user_input])
                        pred = int(preds[0])
                        prob_non_hate = float(probs[0][0])
                        prob_hate = float(probs[0][1])
                        confidence = float(probs[0][pred])
                        model_display = "XLM-RoBERTa Base"
                    elif "Logistic Regression" in selected_model_name:
                        current_model = load_unified_tfidf_lr()
                        sample_series = pd.Series([user_input])
                        pred = int(current_model.predict(sample_series)[0])
                        probs = current_model.predict_proba(sample_series)[0]
                        prob_non_hate = float(probs[0])
                        prob_hate = float(probs[1])
                        confidence = float(probs[pred])
                        model_display = "TF-IDF + Logistic Regression"
                    else:
                        current_model = load_unified_tfidf_svm()
                        sample_series = pd.Series([user_input])
                        pred = int(current_model.predict(sample_series)[0])
                        probs = current_model.predict_proba(sample_series)[0]
                        prob_non_hate = float(probs[0])
                        prob_hate = float(probs[1])
                        confidence = float(probs[pred])
                        model_display = "TF-IDF + Linear SVM"

                except Exception as exc:
                    st.error(f"Inference execution failed: {exc}")
                    st.stop()

            st.markdown("<div style='height:1.25rem;'></div>", unsafe_allow_html=True)

            badge_class = "result-badge-hate" if pred == 1 else "result-badge-nothate"
            badge_text = "HATE" if pred == 1 else "NON-HATE"
            conf_percent = confidence * 100
            nh_percent = prob_non_hate * 100
            h_percent = prob_hate * 100

            res_card_html = f"""
            <div class='ui-card'>
                <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 1.25rem;'>
                    <div class='ui-card-header' style='margin: 0;'>Inference Result</div>
                    <div style='font-size: 0.8rem; color: #818cf8; font-weight: 500;'>Model: {model_display}</div>
                </div>
                <div style='display: flex; gap: 3rem; align-items: center; margin-bottom: 1.5rem;'>
                    <div>
                        <div style='font-size: 0.75rem; color: #94a3b8; text-transform: uppercase; font-weight: 600; margin-bottom: 0.4rem;'>Classification</div>
                        <div class='{badge_class}'>
                            {badge_text}
                        </div>
                    </div>
                    <div>
                        <div style='font-size: 0.75rem; color: #94a3b8; text-transform: uppercase; font-weight: 600; margin-bottom: 0.4rem;'>Model Confidence</div>
                        <div class='conf-val'>{conf_percent:.2f}%</div>
                    </div>
                </div>
                <div style='font-size: 0.75rem; color: #94a3b8; text-transform: uppercase; font-weight: 600; margin-bottom: 0.6rem;'>Class Probabilities</div>
                <div class='prob-item'>
                    <div class='prob-header'>
                        <span>Non-Hate</span>
                        <span>{nh_percent:.2f}%</span>
                    </div>
                    <div class='prob-track'>
                        <div class='prob-fill-nothate' style='width: {nh_percent:.1f}%;'></div>
                    </div>
                </div>
                <div class='prob-item'>
                    <div class='prob-header'>
                        <span>Hate</span>
                        <span>{h_percent:.2f}%</span>
                    </div>
                    <div class='prob-track'>
                        <div class='prob-fill-hate' style='width: {h_percent:.1f}%;'></div>
                    </div>
                </div>
            </div>
            """
            st.markdown(res_card_html, unsafe_allow_html=True)

            # ----------------------------------------------------------
            # SHAP Token Attribution — XLM-R only, on demand
            # ----------------------------------------------------------
            if "XLM-RoBERTa" in selected_model_name:
                st.markdown(
                    "<div style='height:0.75rem;'></div>"
                    "<div style='font-size:1.0rem;font-weight:600;color:#ffffff;"
                    "margin-bottom:0.25rem;'>SHAP Token Attribution</div>"
                    "<div style='font-size:0.8rem;color:#94a3b8;margin-bottom:0.6rem;'>"
                    "Token-level contribution toward the Hate classification "
                    "(computed live from the actual model). "
                    "<span style='color:#f43f5e;font-weight:600;'>Red</span>"
                    " pushes toward Hate &nbsp;|&nbsp; "
                    "<span style='color:#10b981;font-weight:600;'>Green</span>"
                    " pushes toward Non-Hate."
                    "</div>",
                    unsafe_allow_html=True,
                )

                _shap_ok     = False
                _shap_result = None

                with st.spinner(
                    "\u23f3 Computing SHAP token attribution "
                    "\u2014 this takes 15\u201330 seconds..."
                ):
                    try:
                        _shap_exp    = load_shap_explainer()
                        _shap_result = _shap_exp.explain_text(
                            user_input, max_evals=300
                        )
                        _shap_ok = True
                    except Exception as _shap_exc:
                        st.error(
                            f"**SHAP explanation failed:** {_shap_exc}\n\n"
                            "The prediction result above is still valid. "
                            "SHAP is a separate post-hoc computation and does "
                            "not affect model predictions."
                        )

                if _shap_ok and _shap_result:
                    _render_shap_chart(_shap_result, top_k=12)

                    # Token attribution table: top positive + top negative
                    _token_pairs = sorted(
                        zip(_shap_result["tokens"],
                            _shap_result["shap_values_hof"]),
                        key=lambda x: x[1], reverse=True,
                    )
                    _top_pos = [
                        (t.strip() or "\u00b7", v)
                        for t, v in _token_pairs if v > 0
                    ][:5]
                    _top_neg = sorted(
                        [
                            (t.strip() or "\u00b7", v)
                            for t, v in _token_pairs if v < 0
                        ],
                        key=lambda x: x[1],
                    )[:5]

                    _rows_pos = "".join(
                        f"<tr>"
                        f"<td><code style='background:#1e293b;color:#f1f5f9;"
                        f"padding:1px 6px;border-radius:3px;'>{html.escape(t)}</code></td>"
                        f"<td style='color:#f43f5e;font-weight:600;"
                        f"text-align:right;font-family:monospace;'>{v:+.4f}</td>"
                        f"<td style='color:#64748b;font-size:0.78rem;'>&#8594; Hate</td>"
                        f"</tr>"
                        for t, v in _top_pos
                    )
                    _rows_neg = "".join(
                        f"<tr>"
                        f"<td><code style='background:#1e293b;color:#f1f5f9;"
                        f"padding:1px 6px;border-radius:3px;'>{html.escape(t)}</code></td>"
                        f"<td style='color:#10b981;font-weight:600;"
                        f"text-align:right;font-family:monospace;'>{v:+.4f}</td>"
                        f"<td style='color:#64748b;font-size:0.78rem;'>&#8594; Non-Hate</td>"
                        f"</tr>"
                        for t, v in _top_neg
                    )
                    _empty_shap_row = (
                        "<tr><td colspan='3' style='text-align:center;color:#94a3b8;font-size:0.82rem;padding:0.75rem;'>"
                        "No significant token contributions found.</td></tr>"
                    ) if not _rows_pos and not _rows_neg else ""

                    _n_tokens = len(_shap_result["tokens"])
                    _shap_card_html = (
                        "<div class='ui-card' style='margin-top:0.5rem;'>"
                        "<div class='ui-card-header'>Top Token Contributions</div>"
                        "<table class='custom-table'>"
                        "<thead><tr>"
                        "<th>Token</th>"
                        "<th style='text-align:right;'>SHAP Value</th>"
                        "<th>Direction</th>"
                        "</tr></thead>"
                        f"<tbody>{_rows_pos}{_rows_neg}{_empty_shap_row}</tbody>"
                        "</table>"
                        "<div style='font-size:0.78rem;color:#94a3b8;margin-top:0.85rem;line-height:1.6;border-top:1px solid #1e293b;padding-top:0.65rem;'>"
                        f"<span>Base P(Hate): <strong style='color:#f1f5f9;'>{_shap_result['base_value_hof']:.4f}</strong></span>"
                        f" &bull; <span>Model P(Hate): <strong style='color:#f1f5f9;'>{prob_hate:.4f}</strong></span>"
                        f" &bull; <span>Tokens Evaluated: <strong style='color:#f1f5f9;'>{_n_tokens}</strong></span>"
                        "<div style='color:#64748b;font-size:0.74rem;margin-top:0.35rem;'>"
                        "SHAP uses word-boundary masking (\\W+ split). "
                        "Indian-script glyphs may render as boxes in the chart (font fallback only &mdash; attribution values are exact)."
                        "</div>"
                        "</div>"
                        "</div>"
                    )
                    st.markdown(_shap_card_html, unsafe_allow_html=True)

                # ----------------------------------------------------------
                # LIME Feature Attribution — XLM-R only, on demand
                # ----------------------------------------------------------
                st.markdown(
                    "<div style='height:1.25rem;'></div>"
                    "<div style='font-size:1.0rem;font-weight:600;color:#ffffff;"
                    "margin-bottom:0.25rem;'>LIME Feature Attribution (Local Surrogate)</div>"
                    "<div style='font-size:0.8rem;color:#94a3b8;margin-bottom:0.6rem;'>"
                    "Local interpretable linear surrogate model estimating feature contributions around the current input. "
                    "Explained class: <span style='color:#f43f5e;font-weight:600;'>HATE</span>. &nbsp;|&nbsp; "
                    "<span style='color:#f43f5e;font-weight:600;'>Red</span>"
                    " pushes toward Hate &nbsp;|&nbsp; "
                    "<span style='color:#10b981;font-weight:600;'>Green</span>"
                    " pushes away from Hate (toward Non-Hate)."
                    "</div>",
                    unsafe_allow_html=True,
                )

                _lime_ok     = False
                _lime_result = None

                with st.spinner(
                    "\u23f3 Computing LIME feature attribution "
                    "\u2014 evaluating local perturbations (10\u201320s)..."
                ):
                    try:
                        _lime_exp    = load_lime_explainer()
                        _lime_result = _lime_exp.explain_text(
                            user_input,
                            target_class=1,
                            num_features=10,
                            num_samples=250,
                        )
                        _lime_ok = True
                    except Exception as _lime_exc:
                        st.error(
                            f"**LIME explanation failed:** {_lime_exc}\n\n"
                            "The prediction and SHAP results above remain valid. "
                            "LIME is an independent post-hoc surrogate computation."
                        )

                if _lime_ok and _lime_result:
                    _render_lime_chart(_lime_result, top_k=10)

                    # Feature attribution table: positive + negative weights
                    _raw_exp = _lime_result["raw_explanation"]
                    _lime_pos = [(f, w) for f, w in _raw_exp if w > 0][:5]
                    _lime_neg = [(f, w) for f, w in _raw_exp if w < 0][:5]

                    _lime_rows_pos = "".join(
                        f"<tr>"
                        f"<td><code style='background:#1e293b;color:#f1f5f9;"
                        f"padding:1px 6px;border-radius:3px;'>{html.escape(f)}</code></td>"
                        f"<td style='color:#f43f5e;font-weight:600;"
                        f"text-align:right;font-family:monospace;'>{w:+.4f}</td>"
                        f"<td style='color:#64748b;font-size:0.78rem;'>&#8594; Hate</td>"
                        f"</tr>"
                        for f, w in _lime_pos
                    )
                    _lime_rows_neg = "".join(
                        f"<tr>"
                        f"<td><code style='background:#1e293b;color:#f1f5f9;"
                        f"padding:1px 6px;border-radius:3px;'>{html.escape(f)}</code></td>"
                        f"<td style='color:#10b981;font-weight:600;"
                        f"text-align:right;font-family:monospace;'>{w:+.4f}</td>"
                        f"<td style='color:#64748b;font-size:0.78rem;'>&#8594; Non-Hate</td>"
                        f"</tr>"
                        for f, w in _lime_neg
                    )
                    _empty_lime_row = (
                        "<tr><td colspan='3' style='text-align:center;color:#94a3b8;font-size:0.82rem;padding:0.75rem;'>"
                        "No significant feature weights found.</td></tr>"
                    ) if not _lime_rows_pos and not _lime_rows_neg else ""

                    _lime_card_html = (
                        "<div class='ui-card' style='margin-top:0.5rem;'>"
                        "<div class='ui-card-header'>Top LIME Feature Weights (Explained Class: HATE)</div>"
                        "<table class='custom-table'>"
                        "<thead><tr>"
                        "<th>Feature / Word</th>"
                        "<th style='text-align:right;'>LIME Weight</th>"
                        "<th>Direction</th>"
                        "</tr></thead>"
                        f"<tbody>{_lime_rows_pos}{_lime_rows_neg}{_empty_lime_row}</tbody>"
                        "</table>"
                        "<div style='font-size:0.78rem;color:#94a3b8;margin-top:0.85rem;line-height:1.6;border-top:1px solid #1e293b;padding-top:0.65rem;'>"
                        "<span>Explained Class: <strong style='color:#f43f5e;'>HATE</strong></span>"
                        f" &bull; <span>Local Surrogate R&sup2;: <strong style='color:#f1f5f9;'>{_lime_result['score']:.4f}</strong></span>"
                        f" &bull; <span>Perturbation Samples: <strong style='color:#f1f5f9;'>{_lime_result['num_samples']}</strong></span>"
                        "<div style='color:#64748b;font-size:0.74rem;margin-top:0.35rem;'>"
                        "LIME models local feature sensitivity around the current instance via a sparse linear surrogate. "
                        "Surrogate R&sup2; measures local approximation fit, not overall model accuracy, and does not establish causal importance."
                        "</div>"
                        "</div>"
                        "</div>"
                    )
                    st.markdown(_lime_card_html, unsafe_allow_html=True)


# ============================================================
# TAB 2 — PERFORMANCE
# ============================================================

with tab_perf:
    st.markdown(
        "<div style='font-size: 0.85rem; color: #94a3b8; margin-bottom: 1.5rem;'>"
        "Evaluation results on the held-out unified test dataset (n = 6,319 samples). "
        "All metrics reflect verified evaluation outputs."
        "</div>",
        unsafe_allow_html=True,
    )

    if not XLMR_METRICS or not _is_unified_test(XLMR_METRICS):
        st.error(f"Verified XLM-R metrics artifact not found or invalid: {XLMR_METRICS_PATH}")
        st.stop()

    acc = XLMR_METRICS["accuracy"] * 100
    f1 = XLMR_METRICS["macro_f1"] * 100
    auc = XLMR_METRICS["roc_auc"] * 100
    prec = XLMR_METRICS["macro_precision"] * 100
    rec = XLMR_METRICS["macro_recall"] * 100

    if "confusion_matrix" not in XLMR_METRICS or not XLMR_METRICS["confusion_matrix"]:
        st.error("Confusion matrix missing in verified XLM-R metrics artifact.")
        st.stop()
    cm_vals = XLMR_METRICS["confusion_matrix"]

    # 3 Primary Metric Cards
    col_m1, col_m2, col_m3 = st.columns(3)
    with col_m1:
        st.markdown(
            f"<div class='metric-card'>"
            f"<div class='metric-title'>Accuracy (XLM-R)</div>"
            f"<div class='metric-number'>{acc:.2f}%</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
    with col_m2:
        st.markdown(
            f"<div class='metric-card'>"
            f"<div class='metric-title'>Macro F1 (XLM-R)</div>"
            f"<div class='metric-number'>{f1:.2f}%</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
    with col_m3:
        st.markdown(
            f"<div class='metric-card'>"
            f"<div class='metric-title'>ROC-AUC (XLM-R)</div>"
            f"<div class='metric-number'>{auc:.2f}%</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height: 0.75rem;'></div>", unsafe_allow_html=True)

    st.markdown(
        "<div style='font-size: 1.05rem; font-weight: 600; color: #ffffff; margin-bottom: 0.25rem;'>"
        "Secondary Metrics"
        "</div>"
        "<div style='font-size: 0.82rem; color: #94a3b8; margin-bottom: 0.9rem;'>"
        "Macro Precision and Recall"
        "</div>",
        unsafe_allow_html=True,
    )
    c_p1, c_p2 = st.columns(2)
    with c_p1:
        st.markdown(
            f"<div class='metric-card'>"
            f"<div class='metric-title'>Macro Precision (XLM-R)</div>"
            f"<div class='metric-number'>{prec:.2f}%</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
    with c_p2:
        st.markdown(
            f"<div class='metric-card'>"
            f"<div class='metric-title'>Macro Recall (XLM-R)</div>"
            f"<div class='metric-number'>{rec:.2f}%</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)

    # Confusion Matrix Section
    st.markdown(
        "<div style='font-size: 1.05rem; font-weight: 600; color: #ffffff; margin-bottom: 0.35rem;'>"
        "XLM-RoBERTa Test Set Confusion Matrix"
        "</div>"
        "<div style='font-size: 0.8rem; color: #94a3b8; margin-bottom: 1rem;'>"
        "Rows represent actual ground-truth classes; columns represent predicted classes."
        "</div>",
        unsafe_allow_html=True,
    )

    cm_fig_col, cm_tbl_col = st.columns([1.1, 1])

    with cm_fig_col:
        if XLMR_CM_FIG.exists():
            st.image(str(XLMR_CM_FIG), use_container_width=True)
        else:
            st.error(f"Verified confusion matrix figure not found: {XLMR_CM_FIG.name}")

    with cm_tbl_col:
        tn, fp = cm_vals[0][0], cm_vals[0][1]
        fn, tp = cm_vals[1][0], cm_vals[1][1]
        total = tn + fp + fn + tp

        cm_table_html = f"""
        <table class='custom-table'>
            <tr>
                <th>Actual Class</th>
                <th>Predicted Non-Hate</th>
                <th>Predicted Hate</th>
            </tr>
            <tr>
                <td><b>Actual Non-Hate</b></td>
                <td><b>{tn:,}</b> ({tn/total*100:.1f}%)</td>
                <td>{fp:,} ({fp/total*100:.1f}%)</td>
            </tr>
            <tr>
                <td><b>Actual Hate</b></td>
                <td>{fn:,} ({fn/total*100:.1f}%)</td>
                <td><b>{tp:,}</b> ({tp/total*100:.1f}%)</td>
            </tr>
        </table>
        <div style='font-size: 0.8rem; color: #94a3b8; line-height: 1.8; margin-top: 1rem;'>
            True Non-Hate: <b>{tn:,}</b> &nbsp;&bull;&nbsp; False Hate: <b>{fp:,}</b><br>
            False Non-Hate: <b>{fn:,}</b> &nbsp;&bull;&nbsp; True Hate: <b>{tp:,}</b>
        </div>
        """
        st.markdown(cm_table_html, unsafe_allow_html=True)

    st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)

    # Model Comparison Section
    st.markdown(
        "<div style='font-size: 1.05rem; font-weight: 600; color: #ffffff; margin-bottom: 0.35rem;'>"
        "Model Comparison &mdash; Unified Test Set"
        "</div>",
        unsafe_allow_html=True,
    )

    if not BASELINES_VERIFIED:
        st.error(f"Verified baseline metrics artifacts not found or invalid: {TFIDF_LR_METRICS_PATH.name} / {TFIDF_SVM_METRICS_PATH.name}")
    else:
        lr_acc = TFIDF_LR_METRICS["accuracy"] * 100
        lr_f1 = TFIDF_LR_METRICS["macro_f1"] * 100
        lr_auc = TFIDF_LR_METRICS["roc_auc"] * 100

        svm_acc = TFIDF_SVM_METRICS["accuracy"] * 100
        svm_f1 = TFIDF_SVM_METRICS["macro_f1"] * 100
        svm_auc = TFIDF_SVM_METRICS["roc_auc"] * 100

        comp_html = f"""
        <table class='custom-table'>
            <tr>
                <th>Model Architecture</th>
                <th>Test Accuracy</th>
                <th>Test Macro F1</th>
                <th>Test ROC-AUC</th>
            </tr>
            <tr>
                <td>TF-IDF + Logistic Regression</td>
                <td>{lr_acc:.2f}%</td>
                <td>{lr_f1:.2f}%</td>
                <td>{lr_auc:.2f}%</td>
            </tr>
            <tr>
                <td>TF-IDF + Linear SVM</td>
                <td>{svm_acc:.2f}%</td>
                <td>{svm_f1:.2f}%</td>
                <td>{svm_auc:.2f}%</td>
            </tr>
            <tr class='highlight'>
                <td>XLM-RoBERTa Base (Selected Model)</td>
                <td>{acc:.2f}%</td>
                <td>{f1:.2f}%</td>
                <td>{auc:.2f}%</td>
            </tr>
        </table>
        <div style='font-size: 0.75rem; color: #64748b; margin-top: 0.6rem;'>
            Evaluated on the 6,319-sample unified test split across Tamil, Malayalam, Hindi, and Telugu.
        </div>
        """
        st.markdown(comp_html, unsafe_allow_html=True)

    # ----------------------------------------------------------
    # Cross-Language Breakdown Section
    # ----------------------------------------------------------
    st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)
    st.markdown(
        "<div style='font-size: 1.05rem; font-weight: 600; color: #ffffff; margin-bottom: 0.35rem;'>"
        "Cross-Language Performance Breakdown &mdash; XLM-RoBERTa"
        "</div>"
        "<div style='font-size: 0.8rem; color: #94a3b8; margin-bottom: 1rem;'>"
        "Language-stratified evaluation on the unified test set across Tamil, Malayalam, Hindi, and Telugu."
        "</div>",
        unsafe_allow_html=True,
    )

    if LANG_METRICS_PATH.exists():
        lang_df = pd.read_csv(LANG_METRICS_PATH)
        lang_rows = "".join(
            f"<tr>"
            f"<td><b>{row['language']}</b></td>"
            f"<td style='text-align:right;'>{int(row['sample_count']):,}</td>"
            f"<td style='text-align:right;'>{int(row['not_hate_count']):,}</td>"
            f"<td style='text-align:right;'>{int(row['hate_count']):,}</td>"
            f"<td style='text-align:right;'>{row['hate_ratio']*100:.1f}%</td>"
            f"<td style='text-align:right;'>{row['accuracy']*100:.2f}%</td>"
            f"<td style='text-align:right;'><b>{row['macro_f1']*100:.2f}%</b></td>"
            f"<td style='text-align:right;'>{row['roc_auc']*100:.2f}%</td>"
            f"<td style='text-align:right;'>{row['FPR']*100:.1f}%</td>"
            f"<td style='text-align:right;'>{row['FNR']*100:.1f}%</td>"
            f"</tr>"
            for _, row in lang_df.iterrows()
        )
        lang_table_html = f"""
        <table class='custom-table'>
            <tr>
                <th>Language</th>
                <th style='text-align:right;'>Samples</th>
                <th style='text-align:right;'>Non-Hate</th>
                <th style='text-align:right;'>Hate</th>
                <th style='text-align:right;'>Hate %</th>
                <th style='text-align:right;'>Accuracy</th>
                <th style='text-align:right;'>Macro F1</th>
                <th style='text-align:right;'>ROC-AUC</th>
                <th style='text-align:right;'>FPR</th>
                <th style='text-align:right;'>FNR</th>
            </tr>
            {lang_rows}
        </table>
        <div style='font-size: 0.75rem; color: #64748b; margin-top: 0.6rem; line-height: 1.6;'>
            Source: <code>results/error_analysis/language_metrics.csv</code> &bull; 
            Note: Malayalam exhibits 96.0% class imbalance (50 Hate out of 1,261), leading to high accuracy (95.48%) but low minority recall (FNR=70.0%, Macro F1=66.07%).
        </div>
        """
        st.markdown(lang_table_html, unsafe_allow_html=True)

        if LANG_CM_FIG.exists():
            st.markdown(
                "<div style='font-size: 1.05rem; font-weight: 600; color: #ffffff; margin-top: 1.5rem; margin-bottom: 0.25rem;'>"
                "Language Confusion Matrices (2×2 Grid)"
                "</div>"
                "<div style='font-size: 0.82rem; color: #94a3b8; margin-bottom: 0.9rem;'>"
                "Confusion matrices for XLM-RoBERTa across Tamil, Malayalam, Hindi, and Telugu."
                "</div>",
                unsafe_allow_html=True,
            )
            st.image(str(LANG_CM_FIG), use_container_width=True)
    else:
        st.info("Per-language metrics artifact not available.")


# ============================================================
# TAB 3 — METHODOLOGY
# ============================================================

with tab_method:
    st.markdown(
        "<div style='font-size: 1.05rem; font-weight: 600; color: #ffffff; margin-bottom: 0.75rem;'>"
        "Dataset Split & Statistics"
        "</div>",
        unsafe_allow_html=True,
    )

    col_s1, col_s2, col_s3, col_s4 = st.columns(4)
    for col, val, lbl in zip(
        [col_s1, col_s2, col_s3, col_s4],
        ["50,525", "6,315", "6,319", "63,159"],
        ["Training Samples", "Validation Samples", "Test Samples", "Total Samples"],
    ):
        with col:
            st.markdown(
                f"<div class='stat-box'>"
                f"<div class='stat-box-val'>{val}</div>"
                f"<div class='stat-box-lbl'>{lbl}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

    st.markdown("<div style='height: 1.25rem;'></div>", unsafe_allow_html=True)

    c_lng, c_tsk = st.columns(2)
    with c_lng:
        st.markdown("<div class='ui-card-header'>Supported Languages</div>", unsafe_allow_html=True)
        st.markdown(
            "<span class='pill-tag'>Tamil</span>"
            "<span class='pill-tag'>Malayalam</span>"
            "<span class='pill-tag'>Hindi</span>"
            "<span class='pill-tag'>Telugu</span>",
            unsafe_allow_html=True,
        )
    with c_tsk:
        st.markdown("<div class='ui-card-header'>Classification Task</div>", unsafe_allow_html=True)
        st.markdown(
            "<span class='pill-tag'>Binary</span>"
            "<span class='pill-tag'>0 = Non-Hate</span>"
            "<span class='pill-tag'>1 = Hate</span>",
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)

    st.markdown(
        "<div style='font-size: 1.05rem; font-weight: 600; color: #ffffff; margin-bottom: 0.5rem;'>"
        "Model & Training Configuration"
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        "<div style='font-size: 0.88rem; color: #cbd5e1; line-height: 1.6; margin-bottom: 1rem;'>"
        "<b>Architecture:</b> XLM-RoBERTa Base (<code>FacebookAI/xlm-roberta-base</code>) &mdash; "
        "a multilingual Transformer encoder with 278M parameters pretrained on 100 languages. "
        "Fine-tuned for binary classification with cross-entropy loss and AdamW optimization."
        "</div>",
        unsafe_allow_html=True,
    )

    h_cols = st.columns(5)
    for col, lbl, val in zip(
        h_cols,
        ["Epochs", "Learning Rate", "Batch Size", "Max Sequence Length", "Weight Decay"],
        ["5", "2e-5", "16", "128", "0.01"],
    ):
        with col:
            st.markdown(
                f"<div class='stat-box'>"
                f"<div class='stat-box-val' style='font-size: 1.15rem;'>{val}</div>"
                f"<div class='stat-box-lbl'>{lbl}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

    st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)

    st.markdown(
        "<div style='font-size: 1.05rem; font-weight: 600; color: #ffffff; margin-bottom: 0.5rem;'>"
        "Pipeline Architecture"
        "</div>"
        "<div style='font-size: 0.88rem; color: #cbd5e1; line-height: 1.8;'>"
        "Input Code-Mixed Text &rarr; XLM-RoBERTa Tokenization (128 max length) &rarr; "
        "Fine-Tuned XLM-RoBERTa Encoder &rarr; [CLS] Representation &rarr; "
        "Linear Classifier &rarr; Softmax Probabilities &rarr; <b>Non-Hate</b> / <b>Hate</b> Prediction"
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)

    st.markdown(
        "<div style='font-size: 1.05rem; font-weight: 600; color: #ffffff; margin-bottom: 0.5rem;'>"
        "Explainability & Scientific Validation Framework"
        "</div>"
        "<div style='font-size: 0.88rem; color: #cbd5e1; line-height: 1.8;'>"
        "&bull; <b>SHAP:</b> PartitionExplainer with non-word boundary text masking to compute Shapley attribution values.<br>"
        "&bull; <b>LIME:</b> Local Interpretable Model-agnostic Explanations fitting linear surrogate models over localized perturbations.<br>"
        "&bull; <b>Faithfulness Evaluation:</b> Deletion-based feature attribution testing across k &in; {1, 3, 5} features, benchmarked against a deterministic random-deletion baseline with 1,000 bootstrap iterations.<br>"
        "&bull; <b>Cross-Language Error Profiling:</b> Comprehensive confusion matrix and error analysis across Tamil, Malayalam, Hindi, and Telugu.<br>"
        "<span style='color:#94a3b8;font-size:0.8rem;'>"
        "Note: Explanation methods reflect local statistical associations and sensitivity of surrogate models; they represent post-hoc attributions rather than biological or causal proof."
        "</span>"
        "</div>",
        unsafe_allow_html=True,
    )
