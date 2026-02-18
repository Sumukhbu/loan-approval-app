# streamlit_app.py
"""
Streamlit app — Local permutation importance (polished)
- Prettified feature names
- Normalized contributions as percentages (sum of abs -> 100%)
- Suggestions grouped (negatives first, positives second)
- Keeps feature engineering, model loading, debug expanders, and width="stretch"
"""
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import traceback
import random
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer

from feature_engineering import add_engineered_features

st.set_page_config(layout="centered", page_title="Loan Approval — Predict & Explain (Local Importance)")

MODEL_PATH = "artifacts_adv/model.joblib"
BACKGROUND_SAMPLE_SIZE = 50
DEBUG_SUGGESTIONS = False

# UI-level known categories to perturb categoricals realistically
UI_EDUCATION_OPTIONS = ["Graduate", "Not Graduate", "Postgraduate", "Unknown"]
UI_SELF_EMPLOYED_OPTIONS = ["Yes", "No"]
UI_CREDIT_HISTORY_OPTIONS = [1.0, 0.0]

SUGGESTION_TEMPLATES = {
    "income": [
        "Consider increasing your documented income or adding a co-applicant with stable income.",
        "Provide additional income proof like salary slips, tax returns, or business income statements."
    ],
    "loan": [
        "Consider reducing the requested loan amount to improve your loan-to-income ratio.",
        "High loan amount relative to income: consider a co-applicant or reducing loan size."
    ],
    "cibil": [
        "Work on improving your CIBIL score by clearing outstanding dues and resolving credit report issues.",
        "A higher CIBIL score would significantly strengthen your loan application."
    ],
    "credit": [
        "Establish or improve your credit history by maintaining regular payments and clearing dues.",
        "Consider building a stronger credit profile before reapplying."
    ],
    "asset": [
        "Provide detailed asset valuation documents to strengthen your collateral claims.",
        "Consider adding more assets as collateral or providing updated valuations."
    ],
    "dependents": [
        "High number of dependents affects your disposable income. Consider adding a co-applicant.",
        "With more dependents, ensure you have sufficient income documentation."
    ],
    "term": [
        "Consider adjusting the loan term to optimize monthly payment obligations.",
        "A longer term can lower EMIs, but increases total interest; discuss options with lender."
    ],
    "employment": [
        "Provide stable employment documentation and proof of regular income.",
        "Self-employed applicants should provide business registration and tax documents."
    ],
    "education": [
        "Highlight professional qualifications and certificates to strengthen profile.",
        "Educational qualification can be helpful — ensure transcripts are ready."
    ]
}

# ---------------- Helpers ----------------
def load_model_from_joblib(path):
    raw = joblib.load(path)
    if isinstance(raw, dict):
        for k in ("pipeline", "model", "estimator", "clf"):
            if k in raw and (isinstance(raw[k], Pipeline) or isinstance(raw[k], BaseEstimator)):
                return raw[k], f"Found estimator under key '{k}'", raw
        for k, v in raw.items():
            if isinstance(v, Pipeline) or isinstance(v, BaseEstimator):
                return v, f"Found estimator under key '{k}'", raw
        return None, f"joblib dict loaded; keys: {list(raw.keys())}", raw
    if isinstance(raw, Pipeline) or isinstance(raw, BaseEstimator):
        return raw, f"Loaded {raw.__class__.__name__}", raw
    return None, f"Unsupported loaded object type: {type(raw)}", raw

def is_tree_model(estimator):
    cls_name = estimator.__class__.__name__
    tree_indicators = ["RandomForest", "GradientBoosting", "HistGradient", "XGB", "LGBM", "CatBoost"]
    if any(ind in cls_name for ind in tree_indicators):
        return True
    if hasattr(estimator, "estimators_") or hasattr(estimator, "trees_") or hasattr(estimator, "booster"):
        return True
    return False

def prettify_feature_name(name: str) -> str:
    """Map engineered/raw feature keys to human-friendly labels."""
    if name is None:
        return ""
    mapping = {
        "ApplicantIncome": "Applicant Income",
        "CoapplicantIncome": "Coapplicant Income",
        "LoanAmount": "Loan Amount",
        "Loan_Amount_Term": "Loan Term",
        "Loan_Amount_Term": "Loan Term",
        "Credit_History": "Credit History",
        "Cibil_Score": "CIBIL Score",
        "Dependents": "Dependents",
        "Education": "Education",
        "Self_Employed": "Self Employed",
        "Residential_Assets_Value": "Residential Assets",
        "Commercial_Assets_Value": "Commercial Assets",
        "Luxury_Assets_Value": "Luxury Assets",
        "Bank_Asset_Value": "Bank Assets",
        "Total_Assets": "Total Assets",
        "Income_per_dependent": "Income per Dependent",
        "Loan_to_Income": "Loan-to-Income Ratio",
        "Asset_to_Income": "Asset-to-Income Ratio",
        "High_Cibil": "High CIBIL Flag",
        "Loan_ID": "Loan ID",
        "LoanAmount_Term": "Loan Term",
        "Loan_Amount": "Loan Amount",
        "LoanAmount": "Loan Amount"
    }
    # direct mapping first
    if name in mapping:
        return mapping[name]
    # fallback: replace underscores and title-case sensible words
    s = str(name)
    s = s.replace("_", " ").replace(".", " ").strip()
    # keep common abbreviations like CIBIL uppercase
    s = " ".join([w.upper() if w.lower() == "cibil" else w.capitalize() for w in s.split()])
    return s

def plot_probability_bar(prob):
    fig, ax = plt.subplots(figsize=(8, 1.2))
    ax.barh([0], [prob], height=0.6, color='green' if prob >= 0.5 else 'red')
    ax.set_xlim(0, 1)
    ax.set_yticks([])
    ax.set_xlabel("Probability of positive class")
    ax.set_xticks(np.linspace(0, 1, 11))
    ax.set_title(f"{prob*100:.2f}%")
    st.pyplot(fig)

def plot_importances_percent(values_percent, labels, title="Local feature importance (percent)", figsize=(8,6)):
    fig, ax = plt.subplots(figsize=figsize)
    order = np.argsort(np.abs(values_percent))[::-1]
    vals = np.array(values_percent)[order]
    labs = np.array(labels)[order]
    y_pos = np.arange(len(vals))
    ax.barh(y_pos, vals, align='center', color=["#2ca02c" if v>0 else "#d62728" for v in vals])
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labs)
    ax.invert_yaxis()
    ax.set_xlabel("Relative importance (%) — positive helps approval")
    ax.set_title(title)
    plt.tight_layout()
    st.pyplot(fig)

def generate_suggestions_from_features(sorted_features, mapped_inputs):
    suggestions = []
    for feat, contrib in sorted_features[:4]:
        feat_l = feat.lower()
        matched = None
        for cat, templates in SUGGESTION_TEMPLATES.items():
            if cat in feat_l:
                matched = random.choice(templates)
                break
        if matched:
            suggestions.append(f"**{prettify_feature_name(feat)}** (impact: {contrib:.2f}%): {matched}")
        else:
            if contrib < 0:
                suggestions.append(f"**{prettify_feature_name(feat)}** (impact: {contrib:.2f}%) seems to reduce approval probability — consider improving or providing documentation.")
            else:
                suggestions.append(f"**{prettify_feature_name(feat)}** (impact: +{contrib:.2f}%) helps approval — ensure you keep documentation ready.")
    return suggestions

def rule_based_reasons(mapped):
    approve_reasons = []
    reject_reasons = []

    # CIBIL
    cibil = mapped.get("Cibil_Score", None)
    if cibil is not None:
        if cibil >= 750:
            approve_reasons.append(f"CIBIL score is {int(cibil)} (good).")
        elif cibil >= 650:
            approve_reasons.append(f"CIBIL score is {int(cibil)} (fair); consider improvements.")
        else:
            reject_reasons.append(f"CIBIL score is {int(cibil)} which is low (<650). Improve credit history.")

    # Credit history
    ch = mapped.get("Credit_History", None)
    if ch is not None:
        if float(ch) >= 1.0:
            approve_reasons.append("Positive credit history present.")
        else:
            reject_reasons.append("No/poor credit history reported.")

    # Loan-to-income
    ai = mapped.get("ApplicantIncome", 0.0) or 0.0
    loan = mapped.get("LoanAmount", 0.0) or 0.0
    if ai and loan:
        lti = loan / ai
        if lti <= 3:
            approve_reasons.append(f"Loan-to-income ratio is low ({lti:.2f}).")
        elif lti <= 10:
            approve_reasons.append(f"Loan-to-income ratio is moderate ({lti:.2f}); acceptable with documentation.")
        else:
            reject_reasons.append(f"Loan-to-income ratio is very high ({lti:.2f}); consider reducing loan amount or adding a co-applicant.")

    # Assets
    assets_total = sum([
        float(mapped.get("Residential_Assets_Value", 0) or 0),
        float(mapped.get("Commercial_Assets_Value", 0) or 0),
        float(mapped.get("Luxury_Assets_Value", 0) or 0),
        float(mapped.get("Bank_Asset_Value", 0) or 0)
    ])
    if assets_total > 0:
        approve_reasons.append(f"Declared assets total ₹{assets_total:,.0f} which can support collateral claims.")

    # Dependents
    deps = mapped.get("Dependents", 0)
    if deps >= 5:
        reject_reasons.append(f"High number of dependents ({deps}) may reduce disposable income.")
    elif deps >= 3:
        reject_reasons.append(f"Dependents: {deps} — consider demonstrating higher income or co-applicant.")

    # Education / employment
    edu = mapped.get("Education", "")
    if edu:
        if "graduate" in edu.lower() or "post" in edu.lower():
            approve_reasons.append(f"Education ({edu}) is a positive indicator.")
    emp = mapped.get("Self_Employed", "")
    if emp:
        if str(emp).lower() == "yes":
            approve_reasons.append("Self-employed applicant — ensure business income documents (ITR) are provided.")
        else:
            approve_reasons.append("Salaried employment status helps with verification (if applicable).")

    return approve_reasons, reject_reasons

# ---------------- UI ----------------
st.title("🏦 Loan Approval — Predict & Explain (Local Importance)")

col1, col2 = st.columns([2, 1])
with col1:
    st.header("📋 Applicant details (raw inputs)")
    no_of_dependents = st.number_input("Dependents", 0, 20, 2)
    income_annum = st.number_input("Annual Income (₹)", 0.0, 1e9, 500000.0, step=1000.0, format="%.2f")
    loan_amount = st.number_input("Loan Amount (₹)", 0.0, 1e9, 14400000.0, step=1000.0, format="%.2f")
    loan_term = st.number_input("Loan Term (years)", 1, 40, 10)
    cibil_score = st.number_input("CIBIL Score", 300, 900, 596)
    residential_assets_value = st.number_input("Residential Assets (₹)", 0.0, 1e9, 5600000.0, step=1000.0)
    commercial_assets_value = st.number_input("Commercial Assets (₹)", 0.0, 1e9, 3700000.0, step=1000.0)
    luxury_assets_value = st.number_input("Luxury Assets (₹)", 0.0, 1e9, 14500000.0, step=1000.0)
    bank_asset_value = st.number_input("Bank Assets (₹)", 0.0, 1e9, 4500000.0, step=1000.0)
    education = st.selectbox("Education", UI_EDUCATION_OPTIONS)
    self_employed = st.selectbox("Self Employed", UI_SELF_EMPLOYED_OPTIONS)
    coapplicant_income = st.number_input("Coapplicant Income (optional)", 0.0, 1e9, 0.0, step=1000.0)
    credit_history = st.selectbox("Credit History (optional)", options=UI_CREDIT_HISTORY_OPTIONS, format_func=lambda x: "Has history" if x==1.0 else "No history", index=0)
    predict_btn = st.button("🔮 Predict", type="primary")

with col2:
    st.header("ℹ️ Quick explanations")
    st.write("- Inputs are mapped to pipeline columns and feature engineering is applied.")
    st.write("- Local permutation-style importance is used (model-agnostic) and normalized to percentages.")
    st.write("- Deterministic rule-based reasons are shown alongside contributor-based suggestions.")

# ---------------- Load model ----------------
try:
    model_obj, model_msg, raw_loaded = load_model_from_joblib(MODEL_PATH)
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

if model_obj is None:
    st.error(f"Could not extract estimator from joblib. Message: {model_msg}")
    if isinstance(raw_loaded, dict):
        st.write("joblib keys:", list(raw_loaded.keys()))
    st.stop()
else:
    st.success(f"✅ Loaded model: {model_msg}")

# ---------------- Predict flow ----------------
if predict_btn:
    with st.spinner("Processing prediction..."):
        mapped = {
            "LoanAmount": float(loan_amount),
            "Loan_Amount_Term": int(loan_term),
            "Dependents": int(no_of_dependents),
            "ApplicantIncome": float(income_annum),
            "CoapplicantIncome": float(coapplicant_income),
            "Education": str(education),
            "Self_Employed": str(self_employed),
            "Credit_History": float(credit_history),
            "Cibil_Score": int(cibil_score),
            "Residential_Assets_Value": float(residential_assets_value),
            "Commercial_Assets_Value": float(commercial_assets_value),
            "Luxury_Assets_Value": float(luxury_assets_value),
            "Bank_Asset_Value": float(bank_asset_value)
        }

        X_raw = pd.DataFrame([mapped])

        # Feature engineering
        try:
            X_fe = add_engineered_features(X_raw)
        except Exception:
            st.error("Feature engineering failed. See traceback:")
            st.text(traceback.format_exc())
            st.stop()

        # Show engineered features
        with st.expander("📊 Show engineered feature row"):
            try:
                display_df = X_fe.iloc[0:1].T.astype(str)
                st.dataframe(display_df, width="stretch")
            except Exception:
                st.write("Could not display engineered row as a dataframe; showing dict instead.")
                st.write(X_fe.iloc[0].to_dict())

        # Align to pipeline.feature_names_in_ if present
        feature_order = getattr(model_obj, "feature_names_in_", None)
        if feature_order is not None:
            for c in feature_order:
                if c not in X_fe.columns:
                    X_fe[c] = 0 if not any(ch.isalpha() for ch in str(c)) else ""
            X_to_pass = X_fe[list(feature_order)]
        else:
            X_to_pass = X_fe.copy()

        # Prediction
        try:
            if hasattr(model_obj, "predict_proba"):
                proba_arr = model_obj.predict_proba(X_to_pass)
                if proba_arr.ndim == 2 and proba_arr.shape[1] >= 2:
                    prob = float(proba_arr[0, 1])
                else:
                    prob = float(np.max(proba_arr, axis=1)[0])
                method_used = "predict_proba"
            else:
                preds = model_obj.predict(X_to_pass)
                prob = float(preds[0])
                method_used = "predict"
        except Exception:
            st.error("Model prediction failed. Pipeline preprocessing likely expects different columns/dtypes.")
            st.text(traceback.format_exc())
            st.stop()

        # Display prediction
        decision = "✅ Approved" if prob >= 0.5 else "❌ Declined"
        st.subheader("🎯 Prediction result")
        if prob >= 0.5:
            st.success(f"**Decision:** **{decision}**")
        else:
            st.error(f"**Decision:** **{decision}**")
        st.markdown(f"**Probability (positive class):** {prob:.4f}")
        plot_probability_bar(prob)

        # ---------------- Local permutation-style importance (improved & normalized) ----------------
        st.subheader("🔍 Local feature importance (permutation-style)")

        try:
            # Raw background: start with X_to_pass (engineered raw columns)
            bg_raw = X_to_pass.copy()

            # Ensure multiple background rows; duplicate if necessary
            if bg_raw.shape[0] < 8:
                bg_raw = pd.concat([X_to_pass] * max(8, BACKGROUND_SAMPLE_SIZE // 8), ignore_index=True)

            # If bg_raw still lacks diversity (all rows identical), synthesize stronger perturbations for numeric columns
            def background_has_low_variance(df, tol=1e-8):
                try:
                    numeric = df.select_dtypes(include=[np.number])
                    if numeric.shape[1] == 0:
                        return True
                    stds = numeric.std(axis=0, ddof=0).fillna(0).abs()
                    return float(stds.sum()) < tol
                except Exception:
                    return True

            if background_has_low_variance(bg_raw):
                N = max(20, BACKGROUND_SAMPLE_SIZE)
                base = X_to_pass.iloc[0].to_dict()
                rows = []
                for i in range(N):
                    row = base.copy()
                    for k, v in base.items():
                        try:
                            if isinstance(v, (int, float)) and not isinstance(v, bool):
                                scale = max(abs(float(v)) * 0.15, 1.0)  # ±15% or min 1
                                perturb = np.random.normal(loc=0.0, scale=scale)
                                newv = float(v) + perturb
                                if 'depend' in k.lower():
                                    newv = int(max(0, min(20, round(newv))))
                                row[k] = newv
                            else:
                                if k == "Education":
                                    opts = [o for o in UI_EDUCATION_OPTIONS if o != base[k]]
                                    row[k] = random.choice(opts) if opts else base[k]
                                elif k == "Self_Employed":
                                    opts = [o for o in UI_SELF_EMPLOYED_OPTIONS if o != base[k]]
                                    row[k] = random.choice(opts) if opts else base[k]
                                elif k == "Credit_History":
                                    opts = [o for o in UI_CREDIT_HISTORY_OPTIONS if o != base[k]]
                                    row[k] = random.choice(opts) if opts else base[k]
                                else:
                                    row[k] = v
                        except Exception:
                            row[k] = v
                    rows.append(row)
                bg_raw = pd.DataFrame(rows)

            # Final background sampling
            n_bg = min(max(8, bg_raw.shape[0]), 200)
            sampled_bg = bg_raw.sample(n=n_bg, replace=True, random_state=42).reset_index(drop=True)

            orig_prob = prob
            raw_cols = list(X_to_pass.columns)
            contribs = []

            for col in raw_cols:
                perturbed = pd.concat([X_to_pass.iloc[[0]].copy() for _ in range(n_bg)], ignore_index=True)

                # Build replacement list
                replacement_vals = []
                for i in range(n_bg):
                    try:
                        val = sampled_bg.loc[i, col]
                    except Exception:
                        val = sampled_bg.iloc[i][col] if col in sampled_bg.columns else None
                    if pd.isna(val):
                        val = X_to_pass.iloc[0][col]
                    replacement_vals.append(val)
                perturbed[col] = replacement_vals

                # Predict
                try:
                    if hasattr(model_obj, "predict_proba"):
                        p_arr = model_obj.predict_proba(perturbed)
                        if p_arr.ndim == 2 and p_arr.shape[1] >= 2:
                            p_vals = p_arr[:, 1]
                        else:
                            p_vals = np.max(p_arr, axis=1)
                    else:
                        p_vals = model_obj.predict(perturbed)
                        p_vals = np.asarray(p_vals, dtype=float)
                except Exception:
                    if DEBUG_SUGGESTIONS:
                        st.write(f"Prediction failed on perturbed set for {col}; using orig_prob fallback.")
                    p_vals = np.array([orig_prob] * n_bg)

                diffs = orig_prob - np.asarray(p_vals)
                mean_signed = float(np.mean(diffs))
                mean_abs = float(np.mean(np.abs(diffs)))
                sign = float(np.sign(mean_signed)) if not np.isclose(mean_signed, 0.0) else (1.0 if mean_abs > 0 else 0.0)
                contrib = sign * mean_abs
                contribs.append((col, contrib))

            # Normalize to percentages (percentage of total absolute importance)
            abs_total = sum(abs(c) for _, c in contribs) or 1.0
            contribs_percent = [(f, (c / abs_total) * 100.0) for f, c in contribs]
            contribs_percent_sorted = sorted(contribs_percent, key=lambda x: abs(x[1]), reverse=True)

            # Prepare labels and values (prettified)
            labels = [prettify_feature_name(f) for f, _ in contribs_percent_sorted]
            values = [v for _, v in contribs_percent_sorted]

            # Plot normalized percent importances
            plot_importances_percent(values, labels, title="Local importance (positive helps approval)")

            # DataFrame display (pretty names + percent)
            contrib_df = pd.DataFrame([(prettify_feature_name(f), round(v, 4)) for f, v in contribs_percent_sorted],
                                      columns=["Feature", "Contribution (%)"])
            st.dataframe(contrib_df, use_container_width=True)


            # ---------- Combine with rule-based reasons ----------
            approve_reasons, reject_reasons = rule_based_reasons(mapped)

            # top pos/neg (percent)
            top_pos = [(prettify_feature_name(f), v) for f, v in contribs_percent_sorted if v > 0][:5]
            top_neg = [(prettify_feature_name(f), v) for f, v in contribs_percent_sorted if v < 0][:5]

            st.subheader("💡 Human-readable reasons & suggestions")

            # Present deterministic reasons and contributor-based reasons
            if prob >= 0.5:
                st.markdown("**Reasons this application is likely to be approved:**")
                for r in approve_reasons:
                    st.markdown(f"- {r}")
                for f, v in top_pos:
                    st.markdown(f"- {f} (+{v:.2f}%) supports approval")

                if reject_reasons or top_neg:
                    st.markdown("**Areas to improve (may hurt approval):**")
                    for r in reject_reasons:
                        st.markdown(f"- {r}")
                    for f, v in top_neg:
                        st.markdown(f"- {f} ({v:.2f}%) reduces approval probability")
            else:
                st.markdown("**Reasons this application may be declined:**")
                for r in reject_reasons:
                    st.markdown(f"- {r}")
                for f, v in top_neg:
                    st.markdown(f"- {f} ({v:.2f}%) reduces approval probability")

                if approve_reasons or top_pos:
                    st.markdown("**Positive aspects that help the application:**")
                    for r in approve_reasons:
                        st.markdown(f"- {r}")
                    for f, v in top_pos:
                        st.markdown(f"- {f} (+{v:.2f}%) supports approval")

            # ---------- Grouped suggestions ----------
            negatives = [(f, v) for f, v in contribs_percent_sorted if v < 0]
            positives = [(f, v) for f, v in contribs_percent_sorted if v > 0]

            if negatives or positives:
                st.subheader("🛠 Suggestions")
                if negatives:
                    st.markdown("**🔴 Fix these first (hurting approval):**")
                    for f, v in sorted(negatives, key=lambda x: abs(x[1]), reverse=True)[:5]:
                        st.markdown(f"- {prettify_feature_name(f)} (impact: {v:.2f}%) → consider improving or providing documentation.")
                if positives:
                    st.markdown("**🟢 These help you (supporting approval):**")
                    for f, v in sorted(positives, key=lambda x: abs(x[1]), reverse=True)[:5]:
                        st.markdown(f"- {prettify_feature_name(f)} (impact: +{v:.2f}%) → ensure documentation is ready.")

            # Also show short list of automatic suggestions derived from top features (textual)
            textual_suggestions = generate_suggestions_from_features(contribs_percent_sorted, mapped)
            if textual_suggestions:
                st.markdown("**Additional suggestions (based on top contributors):**")
                for i, s in enumerate(textual_suggestions, 1):
                    st.markdown(f"{i}. {s}")

        except Exception:
            st.warning("Local importance generation failed; see trace below.")
            st.text(traceback.format_exc())
