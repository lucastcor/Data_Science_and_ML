# 01 · Advertising Sales — Regression Models from First Principles

Predict product sales from advertising spend across three channels (TV, radio, newspaper) and translate the model into a marketing budget recommendation.

## TL;DR

| Model               | Test RMSE | Test R² |
|---------------------|-----------|---------|
| Naive (mean)        | 4.946     | -0.10   |
| OLS (TV only)       | 3.249     | 0.53    |
| OLS (full)          | 2.088     | 0.80    |
| Ridge (α tuned)     | 2.088     | 0.80    |
| **KNN (k=4, tuned)**| **1.325** | **0.92**|

- **Radio** has the largest sales-per-dollar (≈ +0.19 k-units per \$1k spent).
- **Newspaper** is statistically indistinguishable from zero — recommend cutting the budget.
- **KNN** wins on accuracy because it captures the diminishing-returns shape of TV.
- **OLS** is the model to actually deploy if the marketing team needs explicit elasticities.

## What this notebook demonstrates

- Honest EDA with scatter + LOESS-style trend, Pearson correlations, multicollinearity check.
- **From-scratch implementations** of OLS, Ridge and KNN (numpy normal equations, closed-form ridge, brute-force KNN).
- **5-fold cross-validation** to pick `k` for KNN and `α` for Ridge.
- **Residual diagnostics** (residuals vs fitted, histogram, Q-Q plot) to validate OLS assumptions.
- Model leaderboard with RMSE / MAE / R² and a translation into business actions.
- Side-by-side reference to the **scikit-learn pipeline** that would replace the from-scratch code in production.

## How to run

```bash
pip install -r ../../requirements.txt
jupyter lab 01_advertising_regression.ipynb
```

## Files

- `01_advertising_regression.ipynb` — main notebook (executed, with all outputs).
