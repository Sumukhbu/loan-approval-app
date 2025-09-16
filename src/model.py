import joblib
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from preprocess import build_preprocessor_from_df

def build_pipeline(X_sample=None, model_name='rf', random_state=42):
    """
    Build a sklearn Pipeline with a preprocessor constructed from X_sample and a classifier.
    """
    preprocessor = build_preprocessor_from_df(X_sample if X_sample is not None else None)
    if model_name == 'rf':
        clf = RandomForestClassifier(random_state=random_state, n_jobs=-1)
    else:
        clf = DecisionTreeClassifier(random_state=random_state)
    pipeline = Pipeline([
        ('pre', preprocessor),
        ('clf', clf)
    ])
    return pipeline

def hyperparameter_tuning(pipeline, X, y, model_name='rf'):
    """
    Run GridSearchCV on the given pipeline. Returns fitted GridSearchCV object.
    """
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    if model_name == 'rf':
        param_grid = {
            'clf__n_estimators': [100, 200, 300],
            'clf__max_depth': [5, 10, None],
            'clf__min_samples_split': [2, 5]
        }
    else:
        param_grid = {
            'clf__max_depth': [3, 5, 10, None],
            'clf__min_samples_split': [2, 5, 10]
        }
    gs = GridSearchCV(pipeline, param_grid, cv=cv, scoring='accuracy', n_jobs=-1, verbose=1)
    gs.fit(X, y)
    return gs

def evaluate_model(model, X_test, y_test):
    """
    Returns a dict with accuracy, classification_report string, and roc_auc (if prob available).
    """
    preds = model.predict(X_test)
    proba = None
    try:
        proba = model.predict_proba(X_test)[:, 1]
    except Exception:
        proba = None
    acc = accuracy_score(y_test, preds)
    report = classification_report(y_test, preds, digits=4)
    roc = roc_auc_score(y_test, proba) if proba is not None else None
    return {'accuracy': acc, 'report': report, 'roc_auc': roc}

def save_model(obj, path):
    """
    Save a model/pipeline or bundle dict to given path using joblib.
    """
    joblib.dump(obj, path)
