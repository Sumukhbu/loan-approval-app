# src/shap_analysis.py
import argparse
import joblib
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import os

def load_bundle(path):
    return joblib.load(path)

def prepare_features(df_raw, bundle):
    # Drop drop_cols if provided
    drop_cols = bundle.get("drop_cols") or []
    df_raw = df_raw.drop(columns=[c for c in drop_cols if c in df_raw.columns], errors="ignore")
    pre = bundle.get("preprocessor")
    if pre is None:
        # assume df_raw already has model features
        X = df_raw
    else:
        X = pre.transform(df_raw)
        # shap needs numpy or DataFrame; create DataFrame with feature_names if present
        fn = bundle.get("feature_names")
        if fn:
            X = pd.DataFrame(X, columns=fn)
    return X

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    p.add_argument("--train_csv", default=None, help="Path to training csv to sample background rows (recommended)")
    p.add_argument("--bg_samples", type=int, default=200)
    p.add_argument("--out_dir", default="artifacts/shap")
    p.add_argument("--row_index", type=int, default=None, help="If provided, also generate SHAP for this row index (in train_csv)")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    bundle = load_bundle(args.model_path)
    model = bundle.get("model")
    if model is None:
        raise RuntimeError("Model not found in bundle")

    if args.train_csv is None:
        raise RuntimeError("Pass --train_csv path so we can build a representative background sample")

    df = pd.read_csv(args.train_csv)
    # Drop target if present
    # Choose first column with name matching 'loan_status' or 'target' heuristically
    for t in ("target", "loan_status"):
        if t in df.columns:
            df = df.drop(columns=[t])
            break

    X_bg = df.sample(min(args.bg_samples, len(df)), random_state=42)
    X_bg_pre = prepare_features(X_bg, bundle)

    # Prepare a small test set (the same sample)
    X_test_pre = X_bg_pre.copy()

    # Choose SHAP explainer: TreeExplainer if model is tree-based, otherwise KernelExplainer
    try:
        # Try to get underlying estimator for TreeExplainer
        be = getattr(model, "base_estimator", None) or getattr(model, "estimator", None) or model
        explainer = shap.Explainer(be, X_bg_pre)
    except Exception:
        # Fallback to KernelExplainer (slower)
        explainer = shap.KernelExplainer(lambda x: model.predict_proba(x)[:,1], X_bg_pre)

    shap_values = explainer(X_test_pre)

    # Global summary plot
    plt.figure()
    shap.summary_plot(shap_values, X_test_pre, show=False)
    out_global = os.path.join(args.out_dir, "shap_summary.png")
    plt.savefig(out_global, bbox_inches="tight", dpi=150)
    print("Saved SHAP summary to", out_global)

    # If row_index requested, plot waterfall / force plot for that single example
    if args.row_index is not None:
        idx = args.row_index if args.row_index < len(X_test_pre) else 0
        sv = shap_values[idx]
        plt.figure(figsize=(8,4))
        shap.plots.waterfall(sv, show=False)
        out_row = os.path.join(args.out_dir, f"shap_row_{idx}.png")
        plt.savefig(out_row, bbox_inches="tight", dpi=150)
        print("Saved SHAP waterfall for row", idx, "to", out_row)

    print("Done. Outputs in", args.out_dir)

if __name__ == "__main__":
    main()
