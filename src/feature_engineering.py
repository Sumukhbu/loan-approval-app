
import pandas as pd
import numpy as np

def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create advanced engineered features for the loan dataset.

    Features added:
    - Total_Assets: sum of all asset value columns if present
    - Income_per_dependent: ApplicantIncome / (1 + Dependents)
    - Loan_to_Income: LoanAmount / (ApplicantIncome + 1)
    - Asset_to_Income: Total_Assets / (ApplicantIncome + 1)
    - High_Cibil: binary cibil >= 700
    """
    df = df.copy()
    # Ensure numeric conversions
    for col in ['residential_assets_value','commercial_assets_value','luxury_assets_value','bank_asset_value',
                'Residential_Assets_Value','Commercial_Assets_Value','Luxury_Assets_Value','Bank_Asset_Value']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # handle different column name styles
    asset_cols = []
    for name in ['residential_assets_value','Residential_Assets_Value','commercial_assets_value','Commercial_Assets_Value',
                 'luxury_assets_value','Luxury_Assets_Value','bank_asset_value','Bank_Asset_Value']:
        if name in df.columns:
            asset_cols.append(name)

    if asset_cols:
        df['Total_Assets'] = df[asset_cols].sum(axis=1, skipna=True)
    else:
        df['Total_Assets'] = 0.0

    # Income per dependent
    if 'ApplicantIncome' in df.columns:
        df['Income_per_dependent'] = df['ApplicantIncome'] / (1 + df.get('Dependents', 0).replace(0,0).astype(float))
    elif 'income_annum' in df.columns:
        df['ApplicantIncome'] = pd.to_numeric(df['income_annum'], errors='coerce')
        df['Income_per_dependent'] = df['ApplicantIncome'] / (1 + df.get('no_of_dependents', 0).replace(0,0).astype(float))
    else:
        df['Income_per_dependent'] = 0.0

    # Loan to income
    if 'LoanAmount' in df.columns and 'ApplicantIncome' in df.columns:
        # ensure LoanAmount is monthly? assume as provided
        df['Loan_to_Income'] = df['LoanAmount'] / (df['ApplicantIncome'] + 1)
    else:
        df['Loan_to_Income'] = 0.0

    # Asset to income
    if 'Total_Assets' in df.columns and 'ApplicantIncome' in df.columns:
        df['Asset_to_Income'] = df['Total_Assets'] / (df['ApplicantIncome'] + 1)
    else:
        df['Asset_to_Income'] = 0.0

    # High CIBIL
    cibil_cols = [c for c in df.columns if 'Cibil' in c or 'cibil' in c or 'cibil_score'==c]
    if cibil_cols:
        col = cibil_cols[0]
        df['High_Cibil'] = (pd.to_numeric(df[col], errors='coerce') >= 700).astype(int)
    elif 'Cibil_Score' in df.columns:
        df['High_Cibil'] = (pd.to_numeric(df['Cibil_Score'], errors='coerce') >= 700).astype(int)
    else:
        df['High_Cibil'] = 0

    # Fill NaNs with sensible defaults
    df.fillna({'Total_Assets':0.0,'Income_per_dependent':0.0,'Loan_to_Income':0.0,'Asset_to_Income':0.0,'High_Cibil':0}, inplace=True)
    return df
