# scripts/evaluate_debug.py
import os
import sys

# ensure project root is on sys.path so 'src' package can be imported
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import joblib
import pandas as pd
import numpy as np
from src.predict import infer_and_map_columns, create_derived_features
from sklearn.metrics import classification_report, confusion_matrix

MODEL_PATH = "artifacts_adv/model.joblib"
TEST_CSV = "data/test.csv"

def map_label_series(s):
    s2 = s.astype(str).str.strip().str.lower()
    pos = s2.isin(["y","yes","approved","1","true","t","approved"])
    neg = s2.isin(["n","no","rejected","0","false","f","rejected"])
    mapped = pd.Series(np.nan, index=s2.index)
    mapped[pos] = 1
    mapped[neg] = 0
    return s2, mapped

def main():
    print("[INFO] Loading model:", MODEL_PATH)
    m = joblib.load(MODEL_PATH)
    if isinstance(m, dict) and "pipeline" in m:
        pipe = m["pipeline"]
    else:
        pipe = m

    print("[INFO] Loading test CSV:", TEST_CSV)
    df = pd.read_csv(TEST_CSV)
    df.columns = [c.strip() for c in df.columns]

    if "loan_status" not in df.columns:
        raise SystemExit("ERROR: 'loan_status' column missing after stripping headers. Columns: " + ", ".join(df.columns))

    print("\n[STEP] Raw loan_status value counts (first 50):")
    print(df["loan_status"].astype(str).str.strip().value_counts().head(50))

    s_clean, y_mapped = map_label_series(df["loan_status"])
    print("\n[STEP] loan_status (lowercased) unique values (sample):")
    print(pd.Series(s_clean.unique()).head(50).tolist())

    print("\n[STEP] Mapped y_true value counts (NaN = unknown/unmapped):")
    print(y_mapped.value_counts(dropna=False))

    unknown_mask = y_mapped.isna()
    if unknown_mask.any():
        print(f"\n[WARN] {unknown_mask.sum()} rows have unknown/unmapped loan_status values. Sample:")
        print(df.loc[unknown_mask, ["loan_id","loan_status"]].head(20))

    mapped = infer_and_map_columns(df)
    enriched = create_derived_features(mapped)

    if hasattr(pipe, "feature_names_in_"):
        X = enriched[pipe.feature_names_in_]
    else:
        X = enriched

    print("\n[STEP] Running model.predict() ...")
    y_pred = pipe.predict(X)
    print("Predictions distribution:")
    print(pd.Series(y_pred).value_counts())

    if y_mapped.dropna().nunique() >= 2:
        y_true = y_mapped.fillna(0).astype(int)
        print("\nClassification report (unknowns filled as 0):")
        print(classification_report(y_true, y_pred, zero_division=0))
        print("Confusion matrix:")
        print(confusion_matrix(y_true, y_pred))
        mismatch = (y_true != y_pred)
        if mismatch.any():
            print(f"\n[INFO] Showing up to 10 mismatches (loan_id, true, pred):")
            print(df.loc[mismatch, ["loan_id","loan_status"]].head(10))
            display_cols = ["Loan_ID","ApplicantIncome","LoanAmount","Dependents","Cibil_Score","Total_Assets","Loan_to_Income","High_Cibil"]
            display_cols = [c for c in display_cols if c in enriched.columns]
            print(enriched.loc[mismatch, display_cols].head(10))
    else:
        print("\n[WARN] y_true has fewer than 2 classes after mapping; metrics like ROC AUC are invalid.")
        print("Unique mapped y_true values:", y_mapped.dropna().unique())

if __name__ == "__main__":
    main()
