import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

class DependentsCleaner(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        if 'Dependents' in X.columns:
            # Normalize '3+' and coerce to numeric
            X['Dependents'] = X['Dependents'].replace('3+', 3)
            X['Dependents'] = pd.to_numeric(X['Dependents'], errors='coerce')
        # also handle older column name 'no_of_dependents'
        if 'no_of_dependents' in X.columns and 'Dependents' not in X.columns:
            X['Dependents'] = pd.to_numeric(X['no_of_dependents'], errors='coerce')
        return X


def normalize_columns(df):
    """Normalize column names and create derived columns the project expects.

    - Strips whitespace from column names
    - Renames known columns from your CSV to canonical names used in the pipeline
    - Ensures numeric conversion for numeric-like columns
    - Adds CoapplicantIncome=0 if missing
    - Derives Credit_History from Cibil_Score using threshold 650 if Credit_History absent
    """
    df = df.copy()
    # strip whitespace from column names
    df.columns = df.columns.str.strip()

    col_map = {
        'loan_id': 'Loan_ID',
        'no_of_dependents': 'Dependents',
        'dependents': 'Dependents',
        'education': 'Education',
        'self_employed': 'Self_Employed',
        'income_annum': 'ApplicantIncome',
        'applicantincome': 'ApplicantIncome',
        'loan_amount': 'LoanAmount',
        'loanamount': 'LoanAmount',
        'loan_term': 'Loan_Amount_Term',
        'loan_term_months': 'Loan_Amount_Term',
        'cibil_score': 'Cibil_Score',
        'residential_assets_value': 'Residential_Assets_Value',
        'commercial_assets_value': 'Commercial_Assets_Value',
        'luxury_assets_value': 'Luxury_Assets_Value',
        'bank_asset_value': 'Bank_Asset_Value',
        'loan_status': 'Loan_Status',
        'self employed': 'Self_Employed'
    }

    rename_map = {}
    for c in df.columns:
        key = c.strip().lower()
        if key in col_map:
            rename_map[c] = col_map[key]
    if rename_map:
        df = df.rename(columns=rename_map)

    # Convert numeric-like columns to numeric
    for col in [
        'LoanAmount', 'ApplicantIncome', 'Loan_Amount_Term', 'Cibil_Score',
        'Residential_Assets_Value', 'Commercial_Assets_Value', 'Luxury_Assets_Value', 'Bank_Asset_Value'
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # If CoapplicantIncome missing, create it as zeros
    if 'CoapplicantIncome' not in df.columns:
        df['CoapplicantIncome'] = 0.0

    # Create Credit_History from Cibil_Score if present and Credit_History absent
    if 'Cibil_Score' in df.columns and 'Credit_History' not in df.columns:
        df['Credit_History'] = (df['Cibil_Score'] >= 650).astype(int)

    return df


def build_preprocessor_from_df(X):
    """Build a ColumnTransformer dynamically using only the columns present in X."""
    candidate_numeric = [
        'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term',
        'Dependents', 'Cibil_Score', 'Residential_Assets_Value', 'Commercial_Assets_Value',
        'Luxury_Assets_Value', 'Bank_Asset_Value', 'Total_Assets', 'Income_per_dependent',
        'Loan_to_Income', 'Asset_to_Income', 'High_Cibil'
    ]
    candidate_categorical = [
        'Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area', 'Credit_History'
    ]

    numeric_features = [c for c in candidate_numeric if c in X.columns]
    categorical_features = [c for c in candidate_categorical if c in X.columns]

    transformers = []
    if numeric_features:
        num_pipeline = Pipeline([
            ('dependents_clean', DependentsCleaner()),
            ('imputer', SimpleImputer(strategy='median'))
        ])
        transformers.append(('num', num_pipeline, numeric_features))

    if categorical_features:
        # For scikit-learn >=1.2 use sparse_output=False
        ohe_kwargs = {'handle_unknown': 'ignore'}
        try:
            # prefer sparse_output for newer sklearn
            ohe = OneHotEncoder(**ohe_kwargs, sparse_output=False)
        except TypeError:
            ohe = OneHotEncoder(**ohe_kwargs, sparse=False)

        cat_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('ohe', ohe)
        ])
        transformers.append(('cat', cat_pipeline, categorical_features))

    if not transformers:
        return ColumnTransformer([('pass', 'passthrough', X.columns.tolist())], remainder='drop')

    return ColumnTransformer(transformers, remainder='drop')


def load_data(path_or_buffer):
    """Read CSV from a file path or file-like buffer, normalize columns and return DataFrame."""
    # support both file path strings and file-like objects (uploaded)
    if hasattr(path_or_buffer, "read"):
        df = pd.read_csv(path_or_buffer)
    else:
        df = pd.read_csv(path_or_buffer)
    df = normalize_columns(df)
    return df


def split_X_y(df, target_col='Loan_Status'):
    df = df.copy()
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataframe. Columns: {df.columns.tolist()}")
    # Map common textual labels to binary
    if df[target_col].dtype == object:
        s = df[target_col].astype(str).str.strip()
        mapping = {'Approved': 1, 'Rejected': 0, 'Y': 1, 'N': 0, 'approved': 1, 'rejected': 0}
        y = s.map(mapping)
        if y.isnull().any():
            try:
                y = s.astype(int)
            except Exception:
                pass
    else:
        y = df[target_col]
    X = df.drop(columns=[target_col])
    return X, y
