
# Model Card - Loan Approval Prediction

## Model
- Type: Random Forest (optionally calibrated)
- Framework: scikit-learn
- Saved as: joblib bundle containing pipeline and metadata

## Intended use
Predict loan approval risk for small-medium loan decisions and internal prototyping. Not intended for final underwriting without human review.

## Metrics
Training produces evaluation artifacts: classification report, ROC/PR curves, calibration plot, and SHAP summary.

## Limitations
- Trained on provided dataset; distribution shifts will impact performance.
- No demographic fairness analysis included by default.
- Use with caution for high-stakes decisions.

## Contact
Project owner: student / developer
