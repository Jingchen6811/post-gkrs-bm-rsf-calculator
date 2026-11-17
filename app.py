"""Post-GKRS Brain Metastasis Overall Survival Web Calculator (Streamlit Application).

Run locally:
    streamlit run app.py

Deploy to a public URL (Streamlit Community Cloud):
    Push `web_calculator/` (containing `app.py`, `predictor.py`, `rsf_model_bundle.joblib`,
    and `requirements.txt`) to a GitHub repository and select `web_calculator/app.py` as
    the entrypoint at https://share.streamlit.io.
"""
from __future__ import annotations

import inspect
import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from predictor import load_bundle, predict_patient

st.set_page_config(
    page_title="Post-GKRS Brain Metastasis Survival Calculator (RSF)",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    header[data-testid="stHeader"] {display: none !important;}
    div[data-testid="stToolbar"] {display: none !important;}
    #MainMenu {visibility: hidden !important;}
    footer {visibility: hidden !important;}
    .stApp {
        background-color: #F8FAFC;
        font-family: "Arial", "Segoe UI", -apple-system, sans-serif;
    }
    .block-container {
        padding-top: 0.75rem !important;
        padding-bottom: 0.75rem !important;
        padding-left: 1.25rem !important;
        padding-right: 1.25rem !important;
        max-width: 1420px !important;
    }
    .calc-banner {
        background: linear-gradient(90deg, #1E3557 0%, #384D7A 100%);
        color: #FFFFFF;
        padding: 0.72rem 1.15rem;
        border-radius: 8px;
        margin-bottom: 0.70rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
        box-shadow: 0 2px 6px rgba(30, 53, 87, 0.12);
    }
    .calc-banner-title {
        font-size: 1.22rem;
        font-weight: 700;
        letter-spacing: 0.1px;
        margin: 0;
        line-height: 1.25;
    }
    .calc-banner-sub {
        font-size: 0.84rem;
        color: #E2E8F0;
        margin-top: 0.14rem;
    }
    .section-hdr {
        font-size: 0.96rem;
        font-weight: 700;
        color: #1E293B;
        border-bottom: 2px solid #384D7A;
        padding-bottom: 0.28rem;
        margin-bottom: 0.55rem;
    }
    div[data-testid="stWidgetLabel"] p {
        font-size: 0.83rem !important;
        font-weight: 600 !important;
        color: #1E293B !important;
        margin-bottom: 0.1rem !important;
    }
    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div {
        background-color: #FFFFFF !important;
        border-color: #CBD5E1 !important;
        min-height: 36px !important;
        font-size: 0.86rem !important;
    }
    div.stButton > button[kind="primary"] {
        background-color: #384D7A !important;
        border-color: #384D7A !important;
        color: #FFFFFF !important;
        font-weight: 700 !important;
        font-size: 0.86rem !important;
        border-radius: 6px !important;
        min-height: 38px !important;
    }
    div.stButton > button[kind="secondary"] {
        border-color: #CBD5E1 !important;
        color: #334155 !important;
        font-weight: 600 !important;
        font-size: 0.78rem !important;
        min-height: 38px !important;
    }
    .kpi-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 0.58rem;
        margin-bottom: 0.62rem;
    }
    .kpi-card {
        background: #FFFFFF;
        border: 1px solid #CBD5E1;
        border-top: 4px solid #384D7A;
        border-radius: 7px;
        padding: 0.58rem 0.65rem;
        text-align: center;
        box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04);
    }
    .kpi-label {
        font-size: 0.80rem;
        font-weight: 700;
        color: #475569;
        text-transform: uppercase;
        letter-spacing: 0.3px;
    }
    .kpi-val {
        font-size: 1.62rem;
        font-weight: 800;
        color: #1E3557;
        line-height: 1.18;
        margin: 0.15rem 0;
    }
    .kpi-ci {
        font-size: 0.75rem;
        color: #64748B;
        font-weight: 500;
    }
    .risk-card {
        background: #FFFFFF;
        border: 1px solid #CBD5E1;
        border-radius: 7px;
        padding: 0.68rem 0.85rem;
        margin-bottom: 0.58rem;
        box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04);
    }
    .risk-top {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 0.42rem;
    }
    .risk-title {
        font-size: 0.86rem;
        font-weight: 700;
        color: #1E293B;
    }
    .risk-pill-low {
        background: #EBF2FA;
        color: #1E3557;
        border: 1px solid #384D7A;
        padding: 0.18rem 0.58rem;
        border-radius: 999px;
        font-size: 0.77rem;
        font-weight: 700;
    }
    .risk-pill-high {
        background: #FDF2F2;
        color: #991B1B;
        border: 1px solid #C4615A;
        padding: 0.18rem 0.58rem;
        border-radius: 999px;
        font-size: 0.77rem;
        font-weight: 700;
    }
    .meter-track {
        position: relative;
        height: 12px;
        border-radius: 6px;
        background: linear-gradient(90deg, #DBEAFE 0%, #93C5FD 45%, #FCA5A5 75%, #DC2626 100%);
        margin: 0.42rem 0 0.30rem 0;
    }
    .meter-cut {
        position: absolute;
        top: -4px;
        bottom: -4px;
        width: 2px;
        background: #0F172A;
    }
    .meter-dot {
        position: absolute;
        top: -3px;
        width: 18px;
        height: 18px;
        border-radius: 50%;
        background: #1E3557;
        border: 2.5px solid #FFFFFF;
        box-shadow: 0 1px 4px rgba(0,0,0,0.35);
        transform: translateX(-50%);
    }
    .meter-caption {
        display: flex;
        justify-content: space-between;
        font-size: 0.75rem;
        color: #475569;
        font-weight: 600;
    }
    .shap-box {
        background: #FFFFFF;
        border: 1px solid #CBD5E1;
        border-radius: 7px;
        padding: 0.60rem 0.82rem;
        font-size: 0.80rem;
        color: #1E293B;
        line-height: 1.45;
    }
    .shap-row {
        margin: 0.18rem 0;
    }
    .tag-prot {
        color: #1E3557;
        font-weight: 700;
    }
    .tag-risk {
        color: #B33E33;
        font-weight: 700;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_model_bundle():
    return load_bundle()


bundle = get_model_bundle()
presets = bundle["presets"]
egfr_decimals = int(bundle.get("egfr_decimals", 1))


def apply_preset(preset_key: str) -> None:
    vals = presets[preset_key]
    st.session_state["primary_tumor1"] = int(round(vals["primary_tumor1"]))
    st.session_state["lesions_number_bin"] = int(round(vals["lesions_number_bin"]))
    st.session_state["Extracranial_metastasis"] = int(round(vals["Extracranial_metastasis"]))
    st.session_state["bvz_Group"] = int(round(vals["bvz_Group"]))
    st.session_state["Targeted_agents"] = int(round(vals["Targeted_agents"]))
    st.session_state["fibrinogen"] = float(round(vals["fibrinogen"], 2))
    st.session_state["estimate_gfr"] = float(round(vals["estimate_gfr"], egfr_decimals))
    st.session_state["percentage_of_lymphocytes"] = float(round(vals["percentage_of_lymphocytes"], 1))
    st.session_state["aptt_ratio"] = float(round(vals["aptt_ratio"], 2))


if "fibrinogen" not in st.session_state:
    apply_preset("median")

st.markdown(
    """
    <div class="calc-banner">
        <div>
            <div class="calc-banner-title">Post-GKRS Brain Metastasis Overall Survival Calculator</div>
            <div class="calc-banner-sub">Interpretable 9-Predictor Random Survival Forest &bull; Pre-Infusion Baseline Model (T0)</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

col_in, col_out = st.columns([1.28, 1.0], gap="medium")

with col_in:
    st.markdown('<div class="section-hdr">Baseline Clinical &amp; Laboratory Predictors (Pre-Infusion T0)</div>', unsafe_allow_html=True)

    r1c1, r1c2, r1c3 = st.columns(3, gap="small")
    with r1c1:
        primary_tumor1 = st.selectbox(
            "Primary tumor site",
            options=[1, 2, 3],
            format_func=lambda x: {1: "Lung cancer", 2: "Breast cancer", 3: "Other solid tumor"}[x],
            key="primary_tumor1",
        )
    with r1c2:
        lesions_number_bin = st.selectbox(
            "Intracranial metastases",
            options=[0, 1],
            format_func=lambda x: "≤ 3 lesions" if x == 0 else "> 3 lesions",
            key="lesions_number_bin",
        )
    with r1c3:
        Extracranial_metastasis = st.selectbox(
            "Extracranial metastases",
            options=[0, 1],
            format_func=lambda x: "No" if x == 0 else "Yes",
            key="Extracranial_metastasis",
        )

    r2c1, r2c2, r2c3 = st.columns(3, gap="small")
    with r2c1:
        bvz_Group = st.selectbox(
            "Peri-GKRS bevacizumab",
            options=[0, 1],
            format_func=lambda x: "No (GKRS alone)" if x == 0 else "Yes (Next-day IV)",
            key="bvz_Group",
        )
    with r2c2:
        Targeted_agents = st.selectbox(
            "Prior targeted therapy",
            options=[0, 1],
            format_func=lambda x: "No" if x == 0 else "Yes",
            key="Targeted_agents",
        )
    with r2c3:
        fibrinogen = st.number_input(
            "Plasma fibrinogen (g/L)",
            min_value=1.00,
            max_value=8.50,
            step=0.05,
            format="%.2f",
            key="fibrinogen",
        )

    r3c1, r3c2, r3c3 = st.columns(3, gap="small")
    with r3c1:
        estimate_gfr = st.number_input(
            "eGFR (mL/min/1.73 m²)",
            min_value=15.0,
            max_value=150.0,
            step=0.1 if egfr_decimals == 1 else 0.01,
            format=f"%.{egfr_decimals}f",
            key="estimate_gfr",
        )
    with r3c2:
        percentage_of_lymphocytes = st.number_input(
            "Lymphocytes (%)",
            min_value=1.0,
            max_value=65.0,
            step=0.5,
            format="%.1f",
            key="percentage_of_lymphocytes",
        )
    with r3c3:
        aptt_ratio = st.number_input(
            "APTT ratio",
            min_value=0.60,
            max_value=2.20,
            step=0.02,
            format="%.2f",
            key="aptt_ratio",
        )

    _btn_kw = {"width": "stretch"} if "width" in inspect.signature(st.button).parameters else {"use_container_width": True}
    b1, b2, b3 = st.columns([1.45, 1.0, 1.0], gap="small")
    with b1:
        st.button("Calculate Individualized Survival", type="primary", **_btn_kw)
    with b2:
        st.button("Median-Risk Case", type="secondary", on_click=apply_preset, args=("median",), **_btn_kw)
    with b3:
        st.button("High-Risk Case", type="secondary", on_click=apply_preset, args=("highest",), **_btn_kw)

patient_inputs = {
    "fibrinogen": float(fibrinogen),
    "bvz_Group": float(bvz_Group),
    "lesions_number_bin": float(lesions_number_bin),
    "Extracranial_metastasis": float(Extracranial_metastasis),
    "Targeted_agents": float(Targeted_agents),
    "estimate_gfr": float(estimate_gfr),
    "percentage_of_lymphocytes": float(percentage_of_lymphocytes),
    "aptt_ratio": float(aptt_ratio),
    "primary_tumor1": float(primary_tumor1),
}

pred = predict_patient(bundle, patient_inputs)
ms = pred["milestones"]
pred_risk_pct = 100.0 * pred["predicted_risk_36m"]
base_risk_pct = 100.0 * pred["baseline_risk_36m"]
cut_risk_pct = 100.0 * pred["oof_cutoff_36m"]
pill_cls = "risk-pill-high" if pred["is_high_risk"] else "risk-pill-low"
pill_txt = (
    f"Higher-Risk Stratum (≥ {cut_risk_pct:.1f}% cutoff)"
    if pred["is_high_risk"]
    else f"Lower-Risk Stratum (< {cut_risk_pct:.1f}% cutoff)"
)

shap_df = pred["shap_df"]
prot_rows = shap_df[shap_df["SHAP"] < 0].head(3)
elev_rows = shap_df[shap_df["SHAP"] > 0].head(2)
prot_str = "; ".join(f"{r['Clinical_label']}: {r['Formatted_value']} [{r['SHAP']:+.3f}]" for _, r in prot_rows.iterrows()) or "None"
elev_str = "; ".join(f"{r['Clinical_label']}: {r['Formatted_value']} [{r['SHAP']:+.3f}]" for _, r in elev_rows.iterrows()) or "None"

with col_out:
    st.markdown('<div class="section-hdr">Predicted Overall Survival &amp; Individualized Risk Profile</div>', unsafe_allow_html=True)

    st.markdown(
        f"""
        <div class="kpi-grid">
            <div class="kpi-card">
                <div class="kpi-label">12-Month OS</div>
                <div class="kpi-val">{ms[12]['os_pct']:.1f}%</div>
                <div class="kpi-ci">95% CI: {ms[12]['lo95_pct']:.1f}%&ndash;{ms[12]['hi95_pct']:.1f}%</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">24-Month OS</div>
                <div class="kpi-val">{ms[24]['os_pct']:.1f}%</div>
                <div class="kpi-ci">95% CI: {ms[24]['lo95_pct']:.1f}%&ndash;{ms[24]['hi95_pct']:.1f}%</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">36-Month OS</div>
                <div class="kpi-val">{ms[36]['os_pct']:.1f}%</div>
                <div class="kpi-ci">95% CI: {ms[36]['lo95_pct']:.1f}%&ndash;{ms[36]['hi95_pct']:.1f}%</div>
            </div>
        </div>
        <div class="risk-card">
            <div class="risk-top">
                <div class="risk-title">36-Month Mortality Risk: {pred_risk_pct:.1f}% (Baseline {base_risk_pct:.1f}%)</div>
                <div class="{pill_cls}">{pill_txt}</div>
            </div>
            <div class="meter-track">
                <div class="meter-cut" style="left: {cut_risk_pct:.2f}%;"></div>
                <div class="meter-dot" style="left: {np.clip(pred_risk_pct, 2.0, 98.0):.2f}%;"></div>
            </div>
            <div class="meter-caption">
                <span>0% Low risk</span>
                <span>OOF Median Cutoff: {cut_risk_pct:.1f}%</span>
                <span>100% High risk</span>
            </div>
        </div>
        <div class="shap-box">
            <div class="shap-row"><span class="tag-prot">Protective drivers (SHAP):</span> {prot_str}</div>
            <div class="shap-row"><span class="tag-risk">Risk-elevating drivers (SHAP):</span> {elev_str}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with st.expander("Interactive 0–48 Month Survival Trajectory, Full 9-Predictor SHAP Chart & CSV Export", expanded=False):
    p_col1, p_col2 = st.columns(2, gap="medium")
    curve_df = pred["curve_df"]

    with p_col1:
        fig1, ax1 = plt.subplots(figsize=(5.4, 3.3), dpi=160)
        xm = curve_df["Month"].to_numpy(float)
        ax1.fill_between(xm, curve_df["Lower95"], curve_df["Upper95"], step="post", color="#384D7A", alpha=0.16, linewidth=0)
        ax1.step(xm, curve_df["Cohort_KM_OS"], where="post", color="#64748B", ls="--", lw=1.2, label="Full-cohort KM reference")
        ax1.step(xm, curve_df["Predicted_OS"], where="post", color="#384D7A", ls="-", lw=1.8, label="Predicted patient OS (95% CI)")
        for mh in (12, 24, 36):
            ax1.plot([mh], [ms[mh]["os"]], "o", color="#384D7A", markersize=4.5)
        ax1.set_xlim(0, 48)
        ax1.set_ylim(0, 1.03)
        ax1.set_xticks([0, 12, 24, 36, 48])
        ax1.set_xlabel("Months after initial GKRS")
        ax1.set_ylabel("Survival probability")
        ax1.set_title("Individualized 0–48 Month OS Trajectory", loc="left", fontsize=10, fontweight="bold")
        ax1.spines["top"].set_visible(False)
        ax1.spines["right"].set_visible(False)
        ax1.legend(frameon=False, loc="lower left", fontsize=8)
        fig1.tight_layout()
        st.pyplot(fig1)
        plt.close(fig1)

    with p_col2:
        fig2, ax2 = plt.subplots(figsize=(5.4, 3.3), dpi=160)
        s_plot = shap_df.sort_values("Abs_SHAP", ascending=True).reset_index(drop=True)
        y_pos = np.arange(len(s_plot))
        vals = s_plot["SHAP"].to_numpy(float)
        colors = ["#C4615A" if v >= 0 else "#384D7A" for v in vals]
        ax2.axvline(0.0, color="#64748B", ls="--", lw=0.9)
        ax2.barh(y_pos, vals, color=colors, height=0.62)
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels([f"{r['Clinical_label']}: {r['Formatted_value']}" for _, r in s_plot.iterrows()], fontsize=8)
        ax2.set_xlabel("SHAP contribution to 36-month mortality risk")
        ax2.set_title(
            f"Baseline {pred['baseline_risk_36m']:.3f} → Predicted {pred['predicted_risk_36m']:.3f}",
            loc="left",
            fontsize=10,
            fontweight="bold",
        )
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)
        fig2.tight_layout()
        st.pyplot(fig2)
        plt.close(fig2)

    st.download_button(
        label="Download Individualized Survival Curve (CSV)",
        data=curve_df.to_csv(index=False).encode("utf-8"),
        file_name="rsf_individualized_survival_prediction.csv",
        mime="text/csv",
    )
    st.caption(
        "Clinical research decision-support tool derived from the initial GKRS brain metastasis cohort. "
        "Predictions are intended to assist multidisciplinary prognostic communication and require external validation before standalone clinical use."
    )
