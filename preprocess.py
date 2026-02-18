from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd

class DependentsCleaner(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        if "Dependents" in X.columns:
            X["Dependents"] = X["Dependents"].replace("3+", 3)
            X["Dependents"] = pd.to_numeric(X["Dependents"], errors="coerce")
        return X
