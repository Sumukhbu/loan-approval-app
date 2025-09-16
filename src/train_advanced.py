#!/usr/bin/env python3
"""
Advanced training script for loan_approval_project (robust).
Saves bundle with keys: model, preprocessor, feature_names, drop_cols, meta
"""
import os
import argparse
import json
from datetime import datetime
import logging
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

# Optional: Optuna
try:
    import optuna
    _HAS_OPTUNA = True
except Exception:
    _HAS_OPTUNA = False

# Optional: custom feature engineering hook (if you have it)
try:
    from src.feature_engineering import engineer_features
except Exception:
    engineer_features = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ---------------- Compatibility helpers ----------------
def _make_onehot_encoder():
    """Return OneHotEncoder instance compatible with installed sklearn."""
    try:
        from sklearn.preprocessing import OneHotEncoder
        try:
            return OneHotEncoder(handle_unknown="ignore", sparse=False)
        except TypeError:
            try:
                return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
            except TypeError:
                return OneHotEncoder(handle_unknown="ignore")
    except Exception:
        raise

def _make_calibrated_classifier(base_estimator, cv=3):
    """Construct CalibratedClassifierCV in a version-robust way."""
    try:
        # try base_estimator kw
        return CalibratedClassifierCV(base_estimator=base_estimator, cv=cv)
    except TypeError:
        try:
            return CalibratedClassifierCV(estimator=base_estimator, cv=cv)
        except TypeError:
            try:
                return CalibratedClassifierCV(base_estimator, cv=cv)
            except Exception as e:
                raise RuntimeError(
                    "Unable to construct CalibratedClassifierCV with this sklearn. "
                    f"Original error: {e}"
                )

# ---------------- Type inference & preprocessing ----------------
def infer_column_types_and_ids(df, target_col, id_unique_ratio=0.9, id_cardinality_thresh=0.5):
    """
    Determine numeric and categorical columns and detect identifier-like columns to drop.
    Returns: (num_cols, cat_cols, drop_cols)
    """
    n = len(df)
    candidate_cols = [c for c in df.columns if c != target_col]

    drop_cols = []
    num_cols = []
    cat_cols = []

    for c in candidate_cols:
        ser = df[c]
        nunique = ser.nunique(dropna=True)
        uniq_ratio = nunique / n if n > 0 else 0

        # If almost all values unique OR high-cardinality object column -> treat as ID and drop
        if (uniq_ratio >= id_unique_ratio) or (ser.dtype == "object" and (nunique / max(1, n)) >= id_cardinality_thresh):
            drop_cols.append(c)
            continue

        # Try coercion to numeric
        coerced = pd.to_numeric(ser, errors="coerce")
        num_na = coerced.isna().sum()
        if num_na / max(1, n) < 0.2 and coerced.notna().sum() > 0:
            num_cols.append(c)
        else:
            cat_cols.append(c)

    return num_cols, cat_cols, drop_cols

def build_preprocessor(num_cols, cat_cols):
    """Build numeric + categorical preprocessing ColumnTransformer (robust OneHotEncoder)."""
    from sklearn.pipeline import Pipeline
    numeric_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler())
    ])
    from sklearn.impute import SimpleImputer as SI  # already imported, kept for clarity
    from sklearn.preprocessing import OneHotEncoder as OHE

    # Use compatibility wrapper to get encoder instance
    ohe = _make_onehot_encoder()
    categorical_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="__MISSING__")),
        ("ohe", ohe)
    ])
    transformers = []
    if num_cols:
        transformers.append(("num", numeric_pipeline, num_cols))
    if cat_cols:
        transformers.append(("cat", categorical_pipeline, cat_cols))
    preprocessor = ColumnTransformer(transformers, remainder="drop")
    return preprocessor

def get_output_feature_names(preprocessor):
    """
    Try to extract output feature names from a fitted ColumnTransformer.
    Returns list or None.
    """
    try:
        return list(preprocessor.get_feature_names_out())
    except Exception:
        # Manual fallback: iterate transformers_
        names = []
        try:
            for name, transformer, cols in preprocessor.transformers_:
                if name == "remainder":
                    continue
                # If pipeline with OneHotEncoder:
                if hasattr(transformer, "named_steps") and "ohe" in transformer.named_steps:
                    ohe = transformer.named_steps["ohe"]
                    try:
                        out_names = list(ohe.get_feature_names_out(cols))
                    except Exception:
                        # fallback generic
                        out_names = [f"{c}_ohe" for c in cols]
                    names.extend(out_names)
                else:
                    # numeric passthrough -> keep original column names
                    names.extend(list(cols))
            return names if names else None
        except Exception:
            return None

# ---------------- Optuna objective ----------------
def objective_rf(trial, X, y):
    n_estimators = trial.suggest_categorical("n_estimators", [100, 200, 300, 500])
    max_depth = trial.suggest_int("max_depth", 3, 20)
    min_samples_split = trial.suggest_int("min_samples_split", 2, 12)
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        n_jobs=-1,
        random_state=42
    )
    scores = cross_val_score(clf, X, y, cv=3, scoring="accuracy", n_jobs=-1)
    return float(np.mean(scores))

# ---------------- Main ----------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", required=True, help="Path to training CSV")
    p.add_argument("--out_dir", default="artifacts", help="Output directory")
    p.add_argument("--model", choices=["rf", "dt"], default="rf", help="Model type")
    p.add_argument("--optuna", action="store_true", help="Enable Optuna tuning")
    p.add_argument("--trials", type=int, default=50, help="Number of Optuna trials")
    p.add_argument("--model_name", default="model.joblib", help="Filename for saved model")
    p.add_argument("--target_col", default="target", help="Target column name")
    return p.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    log.info("Loading data from %s", args.data_path)
    df = pd.read_csv(args.data_path)

    if args.target_col not in df.columns:
        raise ValueError(f"Target column '{args.target_col}' not found in {args.data_path}. Columns: {list(df.columns)}")

    # optional feature engineering hook
    if engineer_features:
        log.info("Applying custom feature engineering...")
        df = engineer_features(df)

    # infer types and id-like columns
    num_cols, cat_cols, drop_cols = infer_column_types_and_ids(df, args.target_col)
    log.info("Numeric cols: %s", num_cols)
    log.info("Categorical cols: %s", cat_cols)
    if drop_cols:
        log.info("Detected & dropping identifier/high-cardinality cols: %s", drop_cols)

    # Build X, y excluding drop_cols
    X = df.drop(columns=[args.target_col] + drop_cols)
    y = df[args.target_col]

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    # Build and fit preprocessor
    preprocessor = build_preprocessor(num_cols=[c for c in num_cols if c in X_train.columns],
                                      cat_cols=[c for c in cat_cols if c in X_train.columns])
    log.info("Fitting preprocessor...")
    preprocessor.fit(X_train)
    X_train_trans = preprocessor.transform(X_train)

    # Model selection & tuning
    best_params = None
    if args.model == "rf":
        if args.optuna:
            if not _HAS_OPTUNA:
                log.warning("Optuna requested but not available. Install optuna or run without --optuna.")
            else:
                log.info("Running Optuna hyperparameter search (%d trials)...", args.trials)
                study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
                study.optimize(lambda trial: objective_rf(trial, X_train_trans, y_train),
                               n_trials=args.trials, show_progress_bar=True)
                best_params = study.best_params
                log.info("Optuna best params: %s", best_params)
        if best_params is None:
            best_params = {"n_estimators": 200, "max_depth": 12, "min_samples_split": 4}
        clf_base = RandomForestClassifier(**best_params, n_jobs=-1, random_state=42)
    else:
        best_params = {"max_depth": 8, "min_samples_split": 2}
        clf_base = DecisionTreeClassifier(**best_params, random_state=42)

    # Fit base estimator
    log.info("Fitting base estimator...")
    clf_base.fit(X_train_trans, y_train)

    # Calibrate classifier robustly
    log.info("Calibrating classifier with CalibratedClassifierCV (cv=3)...")
    calib = _make_calibrated_classifier(clf_base, cv=3)
    calib.fit(X_train_trans, y_train)  # fit on transformed features

    # Evaluate
    X_test_trans = preprocessor.transform(X_test)
    preds = calib.predict(X_test_trans)
    probs = calib.predict_proba(X_test_trans)[:, 1] if hasattr(calib, "predict_proba") else None

    acc = accuracy_score(y_test, preds)
    try:
        roc = roc_auc_score(y_test, probs) if probs is not None else None
    except Exception:
        roc = None

    log.info("Test accuracy: %.6f", acc)
    if roc is not None:
        log.info("Test ROC AUC: %.6f", roc)
    log.info("Classification report:\n%s", classification_report(y_test, preds))

    # Feature names
    feature_names = get_output_feature_names(preprocessor)
    if feature_names is None:
        feature_names = [f"f{i}" for i in range(X_train_trans.shape[1])]
    log.info("Number of model features: %d", len(feature_names))

    # Save bundle
    bundle = {
        "model": calib,
        "preprocessor": preprocessor,
        "feature_names": feature_names,
        "drop_cols": drop_cols,
        "meta": {
            "training_date": datetime.utcnow().isoformat() + "Z",
            "model_type": args.model,
            "best_params": best_params,
            "test_accuracy": float(acc),
            "test_roc_auc": float(roc) if roc is not None else None
        }
    }

    out_path = os.path.join(args.out_dir, args.model_name)
    joblib.dump(bundle, out_path)
    log.info("Saved model bundle to %s", out_path)

    # metadata json
    meta_path = out_path + ".meta.json"
    with open(meta_path, "w") as fh:
        json.dump(bundle["meta"], fh, indent=2)
    log.info("Saved metadata to %s", meta_path)

if __name__ == "__main__":
    main()
