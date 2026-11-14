"""Standalone inference module for the Post-GKRS Brain Metastasis RSF Web Calculator.

Loads the serialized full-cohort 9-predictor Random Survival Forest bundle
(`rsf_model_bundle.joblib`) and computes:
1. Continuous 0-48 month survival probabilities and 50-replicate conditional bootstrap 95% CIs;
2. Milestone 12-, 24-, and 36-month OS probabilities and 36-month mortality risk stratification;
3. Exact grouped interventional Shapley attributions summing to Predicted_risk - Baseline_risk.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np
import pandas as pd

BUNDLE_PATH = Path(__file__).resolve().parent / "rsf_model_bundle.joblib"

CLINICAL_LABELS: Dict[str, str] = {
    "fibrinogen": "Plasma fibrinogen",
    "bvz_Group": "Peri-GKRS bevacizumab",
    "lesions_number_bin": "Intracranial metastases",
    "Extracranial_metastasis": "Extracranial metastases",
    "Targeted_agents": "Prior targeted therapy",
    "estimate_gfr": "Estimated GFR",
    "percentage_of_lymphocytes": "Lymphocyte percentage",
    "aptt_ratio": "APTT ratio",
    "primary_tumor1": "Primary tumor site",
}


def load_bundle(path: Path | str | None = None) -> Dict[str, Any]:
    target = Path(path) if path is not None else BUNDLE_PATH
    return joblib.load(target)


def format_clinical_value(var: str, val: float) -> str:
    if var == "bvz_Group":
        return "Yes (Next-day IV)" if int(round(val)) == 1 else "No (GKRS alone)"
    if var == "lesions_number_bin":
        return "> 3 lesions" if int(round(val)) == 1 else "≤ 3 lesions"
    if var in {"Extracranial_metastasis", "Targeted_agents"}:
        return "Yes" if int(round(val)) == 1 else "No"
    if var == "primary_tumor1":
        return {1: "Lung cancer", 2: "Breast cancer", 3: "Other solid tumor"}.get(int(round(val)), str(val))
    if var == "fibrinogen":
        return f"{float(val):.2f} g/L"
    if var == "estimate_gfr":
        return f"{float(val):.1f} mL/min/1.73 m²"
    if var == "percentage_of_lymphocytes":
        return f"{float(val):.1f}%"
    if var == "aptt_ratio":
        return f"{float(val):.2f}"
    return f"{float(val):g}"


def encode_dataframe(enc_dict: Dict[str, Any], df: pd.DataFrame) -> np.ndarray:
    cols = []
    for var_name, kind, levels in enc_dict["spec"]:
        if kind == "num":
            cols.append(df[var_name].to_numpy(dtype=float))
        else:
            arr = df[var_name].to_numpy()
            for lv in levels[1:]:
                cols.append((arr == lv).astype(float))
    if not cols:
        return np.empty((len(df), 0), dtype=float)
    raw = np.column_stack(cols)
    scaled = (raw - enc_dict["mean"]) / enc_dict["scale"]
    return scaled[:, enc_dict["keep"]]


def predict_survival_matrix(
    enc_dict: Dict[str, Any],
    estimator: Any,
    df: pd.DataFrame,
    times: np.ndarray,
) -> np.ndarray:
    times_arr = np.asarray(times, dtype=float)
    x = encode_dataframe(enc_dict, df)
    surv_fns = estimator.predict_survival_function(x)
    return np.vstack(
        [np.r_[1.0, fn.y][np.searchsorted(fn.x, times_arr, side="right")] for fn in surv_fns]
    )


def predict_patient(bundle: Dict[str, Any], inputs: Dict[str, float]) -> Dict[str, Any]:
    variables = list(bundle["variables"])
    row_df = pd.DataFrame([{v: float(inputs[v]) for v in variables}])

    times_grid = np.asarray(bundle["times_grid"], dtype=float)
    surv_main = predict_survival_matrix(bundle["main_encoder"], bundle["main_estimator"], row_df, times_grid)[0]

    boot_curves = np.vstack(
        [
            predict_survival_matrix(b_enc, b_est, row_df, times_grid)[0]
            for b_enc, b_est in bundle["bootstrap_models"]
        ]
    )
    surv_lo95, surv_hi95 = np.quantile(boot_curves, [0.025, 0.975], axis=0)

    milestones = {}
    for m in (12, 24, 36):
        idx_m = int(np.flatnonzero(np.isclose(times_grid, float(m)))[0])
        milestones[m] = {
            "os": float(surv_main[idx_m]),
            "os_pct": float(100.0 * surv_main[idx_m]),
            "lo95_pct": float(100.0 * surv_lo95[idx_m]),
            "hi95_pct": float(100.0 * surv_hi95[idx_m]),
        }

    horizon = float(bundle["horizon_months"])
    pred_risk_36m = float(
        1.0 - predict_survival_matrix(bundle["main_encoder"], bundle["main_estimator"], row_df, np.array([horizon]))[0, 0]
    )

    # Exact batched interventional Shapley decomposition over stored Monte Carlo draws
    bg_draws = bundle["shap_bg_draws"]
    bg_preds = np.asarray(bundle["shap_bg_preds"], dtype=float)
    orders = np.asarray(bundle["shap_orders"], dtype=int)
    n_perms, p_vars = orders.shape

    state_rows = []
    for m_idx in range(n_perms):
        z = dict(bg_draws[m_idx])
        for j_idx in orders[m_idx]:
            v_name = variables[int(j_idx)]
            z[v_name] = float(inputs[v_name])
            state_rows.append(dict(z))

    states_df = pd.DataFrame(state_rows, columns=variables)
    after_surv = predict_survival_matrix(
        bundle["main_encoder"],
        bundle["main_estimator"],
        states_df,
        np.array([horizon]),
    )[:, 0]
    after_risk = (1.0 - after_surv).reshape(n_perms, p_vars)

    shap_vals = np.zeros(p_vars, dtype=float)
    for m_idx in range(n_perms):
        before_val = float(bg_preds[m_idx])
        for k_pos, j_idx in enumerate(orders[m_idx]):
            cur_val = float(after_risk[m_idx, k_pos])
            shap_vals[int(j_idx)] += cur_val - before_val
            before_val = cur_val
    shap_vals /= float(n_perms)

    baseline_risk = float(np.mean(bg_preds))
    cutoff = float(bundle["oof_median_cutoff_36m"])
    is_high_risk = pred_risk_36m >= cutoff

    shap_records = []
    for j_idx, v_name in enumerate(variables):
        shap_records.append(
            {
                "Variable": v_name,
                "Clinical_label": CLINICAL_LABELS.get(v_name, v_name),
                "Raw_value": float(inputs[v_name]),
                "Formatted_value": format_clinical_value(v_name, float(inputs[v_name])),
                "SHAP": float(shap_vals[j_idx]),
                "Abs_SHAP": float(abs(shap_vals[j_idx])),
            }
        )
    shap_df = pd.DataFrame(shap_records).sort_values("Abs_SHAP", ascending=False).reset_index(drop=True)

    curve_df = pd.DataFrame(
        {
            "Month": times_grid,
            "Predicted_OS": surv_main,
            "Predicted_OS_pct": np.round(100.0 * surv_main, 1),
            "Lower95": surv_lo95,
            "Upper95": surv_hi95,
            "Cohort_KM_OS": np.asarray(bundle["cohort_km_ref"], dtype=float),
        }
    )

    return {
        "milestones": milestones,
        "predicted_risk_36m": pred_risk_36m,
        "baseline_risk_36m": baseline_risk,
        "oof_cutoff_36m": cutoff,
        "risk_stratum": "Higher-Risk Stratum" if is_high_risk else "Lower-Risk Stratum",
        "is_high_risk": bool(is_high_risk),
        "shap_df": shap_df,
        "curve_df": curve_df,
    }
