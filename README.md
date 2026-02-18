⚠️ IMPORTANT

This project requires:
- Python 3.9
- Exact dependencies in requirements.txt

It will NOT work on Python 3.12+ due to scikit-learn pickle compatibility.

# Loan Approval — Predict & Explain (Local Importance)

A Streamlit app for loan approval prediction with **local, model-agnostic explanations** (permutation-style importance) and **human-readable reasons & suggestions**.

This repo **includes the trained model binary** (`artifacts_adv/model.joblib`) so you can run it out of the box.

---

## Repo structure
loan-approval-app/
│
├── app/
│   ├── streamlit_app.py
│   └── __init__.py
│
├── artifacts/
├── artifacts_adv/
│   └── model.joblib
│
├── data/
├── scripts/
├── src/
├── tests/
│
├── feature_engineering.py
├── preprocess.py
├── requirements.txt
├── docker-compose.yml
├── Dockerfile.api
├── README.md
├── MODEL_CARD.md
├── metrics.json
├── predictions.csv
├── runtime.txt


