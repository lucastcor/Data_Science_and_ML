# 03 · Telco Customer Churn — End-to-End Predictive Pipeline

Full predictive workflow on a 10 000-row synthetic Telco churn dataset, structured the way a Data Scientist is actually expected to ship: business framing → data → feature engineering → multi-model comparison → P&L-driven threshold → interpretability → versioned artifact.

## TL;DR

| Metric                                  | Value     |
|-----------------------------------------|-----------|
| Test PR-AUC                             | **0.597** |
| Test ROC-AUC                            | **0.813** |
| Chosen decision threshold               | 0.13      |
| Recall at chosen threshold              | 98.3%     |
| Precision at chosen threshold           | 30.7%     |
| Estimated net retained revenue (test)   | **\$192 398** |
| **Uplift vs default 0.5 threshold**     | **+\$87 272** |

Choosing the threshold from the P&L curve — not from the default 0.5 — almost doubled the retained revenue on the held-out test set.

## What this notebook demonstrates

- **Business framing first.** Per-decision economics (TP / FP / FN / TN dollar values) computed before any model is fitted.
- **Feature engineering** — one-hot encoding, `charges_per_tenure` ratio, tenure bucket, hand-crafted interactions (`tenure × month-to-month`, `charges × fibre`).
- **Multi-model comparison** with 5-fold stratified CV using PR-AUC as the optimisation target: logistic, logistic + interactions, KNN at multiple `k`.
- **Hyperparameter tuning** for the L2 regularisation strength.
- **Threshold selection from a P&L curve**, not from default 0.5 — single most impactful decision in the notebook.
- **Permutation feature importance** — model-agnostic, reflects what the model actually uses.
- **Calibration analysis** with a reliability diagram + a note on Platt/isotonic recalibration.
- **Versioned model artifact** saved with the scaler statistics, coefficients, intercept, threshold and feature names.
- **Production checklist** covering MLflow tracking, FastAPI serving, drift monitoring, retraining cadence, and fairness audits.
- **scikit-learn equivalent** snippet at the bottom so the reader can map the from-scratch code to the production library.

## How to run

```bash
pip install -r ../../requirements.txt
jupyter lab 03_telco_churn_endtoend.ipynb
```

The notebook is self-contained — the synthetic dataset is generated deterministically (`random_state=42`) inside the notebook itself.

## Files

- `03_telco_churn_endtoend.ipynb` — main notebook.
- `artifacts/churn_model_v1.npz` — serialised scaler statistics, model coefficients, intercept, decision threshold and feature names.
