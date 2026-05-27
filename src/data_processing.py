"""Data loading and small preprocessing helpers shared across notebooks."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = REPO_ROOT / "data" / "raw"


def load_advertising() -> pd.DataFrame:
    """Load the Advertising dataset and drop the unnamed index column."""
    df = pd.read_csv(DATA_RAW / "advertising.csv")
    if df.columns[0].startswith("Unnamed") or df.columns[0] == "":
        df = df.drop(df.columns[0], axis=1)
    return df


def load_lung_cancer() -> pd.DataFrame:
    """Load the lung cancer survey with a clean, consistent 0/1 encoding.

    The CSV in this repo already encodes symptoms as 0/1 (unlike the version
    on Kaggle that uses 1/2). This helper standardises column names, recodes
    `GENDER` and `LUNG_CANCER` to 0/1, and drops the 33 duplicate rows that
    show up in the raw file (309 → 276 rows).
    """
    df = pd.read_csv(DATA_RAW / "survey_lung_cancer.csv")
    df.columns = [c.strip().replace(" ", "_") for c in df.columns]

    df["GENDER"] = df["GENDER"].map({"MALE": 0, "FEMALE": 1}).astype(int)
    df["LUNG_CANCER"] = df["LUNG_CANCER"].map({"YES": 1, "NO": 0}).astype(int)

    df = df.drop_duplicates().reset_index(drop=True)
    return df


def generate_telco_churn(n: int = 10_000, random_state: int = 42) -> pd.DataFrame:
    """Generate a synthetic Telco-style churn dataset.

    The structure mirrors the well-known IBM Telco Customer Churn dataset:
    demographics, account info, monthly charges, contract type, services, and a
    binary `Churn` target with a realistic ~26% positive rate.

    The data-generating process intentionally bakes in a few well-known
    drivers (month-to-month contracts, high monthly charge, low tenure) so the
    downstream model has a real signal to learn — and so the SHAP explanations
    are interesting.
    """
    rng = np.random.default_rng(random_state)

    gender = rng.choice(["Male", "Female"], size=n)
    senior = rng.choice([0, 1], size=n, p=[0.84, 0.16])
    partner = rng.choice(["Yes", "No"], size=n, p=[0.48, 0.52])
    dependents = rng.choice(["Yes", "No"], size=n, p=[0.30, 0.70])
    tenure = rng.integers(0, 73, size=n)

    phone_service = rng.choice(["Yes", "No"], size=n, p=[0.90, 0.10])
    internet_service = rng.choice(
        ["DSL", "Fiber optic", "No"], size=n, p=[0.34, 0.44, 0.22]
    )
    contract = rng.choice(
        ["Month-to-month", "One year", "Two year"], size=n, p=[0.55, 0.21, 0.24]
    )
    paperless_billing = rng.choice(["Yes", "No"], size=n, p=[0.59, 0.41])
    payment_method = rng.choice(
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)",
        ],
        size=n,
        p=[0.34, 0.23, 0.22, 0.21],
    )

    base_charge = np.where(internet_service == "No", 20, 0)
    base_charge = np.where(internet_service == "DSL", 50, base_charge)
    base_charge = np.where(internet_service == "Fiber optic", 80, base_charge)
    monthly_charges = base_charge + rng.normal(loc=15, scale=10, size=n)
    monthly_charges = np.clip(monthly_charges, 18.0, 120.0)
    total_charges = (monthly_charges * np.maximum(tenure, 1)).round(2)

    # Latent churn propensity — bakes in known drivers.
    logit = (
        -1.6
        + 1.5 * (contract == "Month-to-month")
        - 1.1 * (contract == "Two year")
        + 0.9 * (internet_service == "Fiber optic")
        + 0.6 * (payment_method == "Electronic check")
        - 0.03 * tenure
        + 0.015 * (monthly_charges - 60)
        + 0.4 * senior
        - 0.3 * (partner == "Yes")
        - 0.25 * (dependents == "Yes")
        + rng.normal(0, 0.5, size=n)
    )
    churn_prob = 1 / (1 + np.exp(-logit))
    churn = (rng.uniform(size=n) < churn_prob).astype(int)

    df = pd.DataFrame(
        {
            "gender": gender,
            "senior_citizen": senior,
            "partner": partner,
            "dependents": dependents,
            "tenure_months": tenure,
            "phone_service": phone_service,
            "internet_service": internet_service,
            "contract": contract,
            "paperless_billing": paperless_billing,
            "payment_method": payment_method,
            "monthly_charges": monthly_charges.round(2),
            "total_charges": total_charges,
            "churn": churn,
        }
    )
    return df
