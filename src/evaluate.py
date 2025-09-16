# src/evaluate.py
import argparse
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from predict import infer_and_map_columns, create_derived_features  # reuse functions

def main(model_path, test_csv, out_file):
    print(f"[INFO] Loading model from: {model_path}")
    model_obj = joblib.load(model_path)
    if isinstance(model_obj, dict) and "pipeline" in model_obj:
        model_obj = model_obj["pipeline"]

    if not hasattr(model_obj, "predict"):
        raise ValueError("Loaded model does not support predict()")

    print(f"[INFO] Loading test data from: {test_csv}")
    df = pd.read_csv(test_csv)

    # Prepare features and labels
    mapped = infer_and_map_columns(df)
    enriched = create_derived_features(mapped)

    if "loan_status" not in df.columns:
        raise ValueError("Test CSV must contain 'loan_status' as ground truth labels")

    y_true = df["loan_status"].astype(str).str.lower().map(
        lambda x: 1 if x in ["y", "yes", "approved", "1", "true"] else 0
    )

    X = enriched[model_obj.feature_names_in_]

    print("[INFO] Running predictions...")
    y_pred = model_obj.predict(X)

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }

    if hasattr(model_obj, "predict_proba"):
        try:
            y_proba = model_obj.predict_proba(X)[:, 1]
            metrics["roc_auc"] = roc_auc_score(y_true, y_proba)
        except Exception:
            pass

    # Save metrics
    with open(out_file, "w") as f:
        json.dump(metrics, f, indent=4)

    print(f"[OK] Metrics saved to {out_file}")
    print(json.dumps(metrics, indent=4))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--test_csv", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    main(args.model_path, args.test_csv, args.out)
