#!/usr/bin/env python3
"""
Predict script that loads the saved model bundle from train_advanced.py,
drops identifier-like columns saved as `drop_cols`, aligns/validates input columns,
applies the saved preprocessor, and writes predictions with probability (if available).
"""
import argparse
import joblib
import pandas as pd
import os
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True, help="Path to saved model bundle (.joblib)")
    p.add_argument("--input_csv", required=True, help="CSV with raw input rows")
    p.add_argument("--out_csv", default="predictions.csv", help="Output CSV file")
    return p.parse_args()

def get_expected_raw_columns(preprocessor):
    """
    Best-effort: try to extract the raw columns the preprocessor expects (from transformers_).
    Returns list or None.
    """
    try:
        expected = []
        for name, transformer, cols in preprocessor.transformers_:
            if name == "remainder":
                continue
            expected.extend(list(cols))
        return expected
    except Exception:
        return None

def align_and_transform(df_raw, bundle):
    preprocessor = bundle.get("preprocessor")
    feature_names = bundle.get("feature_names")
    drop_cols = bundle.get("drop_cols") or []

    # drop identifier-like columns if present
    for c in list(drop_cols):
        if c in df_raw.columns:
            log.info("Dropping identifier/high-cardinality column from input: %s", c)
            df_raw = df_raw.drop(columns=[c])

    if preprocessor is not None:
        # Check that required raw columns exist
        expected_raw = get_expected_raw_columns(preprocessor)
        if expected_raw:
            missing = [c for c in expected_raw if c not in df_raw.columns]
            if missing:
                raise ValueError(f"Input CSV missing required raw columns for preprocessing: {missing}. "
                                 f"Expected columns (best-effort): {expected_raw}")
        try:
            X_trans = preprocessor.transform(df_raw)
        except Exception as e:
            raise RuntimeError(f"Preprocessor.transform failed: {e}")
        # Build DataFrame using feature_names if present
        if feature_names:
            X_df = pd.DataFrame(X_trans, columns=feature_names, index=df_raw.index)
        else:
            X_df = pd.DataFrame(X_trans, index=df_raw.index)
        return X_df
    else:
        # no preprocessor: align by feature_names if present
        if feature_names:
            missing = [c for c in feature_names if c not in df_raw.columns]
            if missing:
                raise ValueError(f"Input CSV missing required feature columns: {missing}")
            return df_raw[feature_names].copy()
        return df_raw.copy()

def main():
    args = parse_args()
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model bundle not found at {args.model_path}")

    log.info("Loading model bundle from %s", args.model_path)
    bundle = joblib.load(args.model_path)
    model = bundle.get("model")
    if model is None:
        raise RuntimeError("Loaded bundle does not contain 'model' key")

    log.info("Reading input CSV %s", args.input_csv)
    df_in = pd.read_csv(args.input_csv)

    try:
        X_df = align_and_transform(df_in, bundle)
    except Exception as e:
        # augment error with expected raw columns if possible
        extra = ""
        pre = bundle.get("preprocessor")
        if pre is not None:
            exp = get_expected_raw_columns(pre)
            if exp:
                extra = f"\nPreprocessor expected raw columns (best-effort): {exp}"
        raise RuntimeError(f"Failed to align/transform input CSV: {e}{extra}")

    log.info("Running predictions on %d rows", len(X_df))
    preds = model.predict(X_df)
    probs = None
    try:
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_df)[:, 1]
    except Exception:
        log.warning("predict_proba failed; continuing without probabilities")

    out = df_in.copy()
    out["pred"] = preds
    if probs is not None:
        out["prob"] = probs

    out.to_csv(args.out_csv, index=False)
    log.info("Saved predictions to %s", args.out_csv)

if __name__ == "__main__":
    main()
