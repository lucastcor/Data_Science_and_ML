# 02 · Lung Cancer Risk Screening — Logistic Regression & KNN (Imbalanced Class)

Binary classification on a 276-row symptom survey (86% positive class). The notebook is built around the realistic constraint that **a missed cancer case is far more costly than an unnecessary follow-up scan** — so the metric set and the decision threshold are chosen accordingly.

## TL;DR

| Model                           | Accuracy | Precision | Recall | F1   | ROC-AUC | PR-AUC |
|---------------------------------|----------|-----------|--------|------|---------|--------|
| LogReg @ default 0.5            | 0.84     | 0.98      | 0.83   | 0.90 | 0.91    | 0.99   |
| **LogReg @ cost-tuned threshold** | 0.87     | 0.87      | **1.00** | 0.93 | 0.91    | 0.99   |
| KNN (k=11)                      | 0.87     | 0.91      | 0.94   | 0.92 | 0.87    | 0.98   |

- **PR-AUC ≈ 0.99** on a held-out 30% test set.
- Threshold tuned by a cost model where FN is 5× more costly than FP — drives **recall to 1.0** (no missed cases) while keeping precision at 0.87.
- Top odds-ratio drivers: **allergy**, **alcohol consuming**, **swallowing difficulty**.

## What this notebook demonstrates

- **From-scratch logistic regression with gradient descent**, including class-weight balancing and L2 regularisation.
- Stratified 5-fold CV to tune the L2 penalty using **PR-AUC** as the optimisation target (not accuracy — class imbalance makes accuracy misleading).
- ROC + PR curves plotted side by side and analysed together.
- **Cost-based decision-threshold selection** instead of the default 0.5.
- **Calibration plot** (reliability diagram) to check whether predicted probabilities can be trusted as risks.
- Coefficient → **odds-ratio** interpretation on standardised features.
- An explicit **limitations & ethical guardrails** section covering selection bias, label quality, fairness, and decision-support framing.
- Reference snippet showing how the same pipeline looks in **scikit-learn**.

## How to run

```bash
pip install -r ../../requirements.txt
jupyter lab 02_lung_cancer_classification.ipynb
```

## Files

- `02_lung_cancer_classification.ipynb` — main notebook (executed, with all outputs).
