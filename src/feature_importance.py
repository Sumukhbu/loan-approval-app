# src/feature_importance.py
import joblib
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

def extract_feature_importances(model):
    """
    Robust extraction of feature importances from a calibrated / wrapped estimator.
    Returns numpy array or None.
    """
    # 1) try model.base_estimator (CalibratedClassifierCV older APIs)
    try:
        be = getattr(model, "base_estimator", None)
        if be is not None and hasattr(be, "feature_importances_"):
            return np.array(be.feature_importances_)
    except Exception:
        pass

    # 2) try model.estimator (some versions)
    try:
        be = getattr(model, "estimator", None)
        if be is not None and hasattr(be, "feature_importances_"):
            return np.array(be.feature_importances_)
    except Exception:
        pass

    # 3) try model.calibrated_classifiers_ (CalibratedClassifierCV internally stores calibrated classifiers)
    try:
        # calibrated_classifiers_ is a list of fitted estimators (one per class / cv)
        ccs = getattr(model, "calibrated_classifiers_", None)
        if ccs and len(ccs) > 0:
            # take the first underlying estimator if it has feature_importances_
            first = ccs[0]
            # some wrappers store .base_estimator inside each classifier
            be = getattr(first, "base_estimator", None) or getattr(first, "estimator", None) or first
            if hasattr(be, "feature_importances_"):
                return np.array(be.feature_importances_)
    except Exception:
        pass

    # 4) try model.__dict__ scan (last resort)
    for attr in ["feature_importances_", "_clf", "estimator_"]:
        try:
            candidate = getattr(model, attr, None)
            if candidate is not None and hasattr(candidate, "feature_importances_"):
                return np.array(candidate.feature_importances_)
        except Exception:
            pass

    return None

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True, help="Path to joblib bundle (artifacts/model.joblib)")
    p.add_argument("--topk", type=int, default=20)
    p.add_argument("--plot_out", default="feature_importances.png")
    args = p.parse_args()

    bundle = joblib.load(args.model_path)
    feature_names = bundle.get("feature_names")
    model = bundle.get("model")
    if feature_names is None:
        raise RuntimeError("Bundle missing feature_names")
    if model is None:
        raise RuntimeError("Bundle missing model")

    imps = extract_feature_importances(model)
    if imps is None:
        print("Could not extract feature_importances_ from model. The model may not be tree-based.")
        return

    if len(imps) != len(feature_names):
        print("Warning: number of importances != number of feature names. Using min length.")
    L = min(len(imps), len(feature_names))
    pairs = sorted(zip(feature_names[:L], imps[:L]), key=lambda x: x[1], reverse=True)
    topk = pairs[: args.topk]
    print("Top feature importances:")
    for name, val in topk:
        print(f"{name}: {val:.6f}")

    # Plot
    names = [p[0] for p in topk][::-1]
    vals = [p[1] for p in topk][::-1]
    plt.figure(figsize=(8, max(3, 0.3 * len(names))))
    plt.barh(names, vals)
    plt.xlabel("Feature importance")
    plt.tight_layout()
    plt.savefig(args.plot_out, dpi=150)
    print("Saved plot to", args.plot_out)

if __name__ == "__main__":
    main()
