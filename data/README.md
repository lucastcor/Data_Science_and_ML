# Data dictionary

Small public datasets used across the notebooks. Each file lives in `data/raw/` and is read-only.

## `advertising.csv`

Classic dataset from *An Introduction to Statistical Learning* (James, Witten, Hastie, Tibshirani). 200 rows, all numeric. Used in notebook **01** to compare KNN and linear models.

| Column     | Type  | Description                              |
|------------|-------|------------------------------------------|
| TV         | float | TV ad budget (thousands of USD)          |
| radio      | float | Radio ad budget (thousands of USD)       |
| newspaper  | float | Newspaper ad budget (thousands of USD)   |
| sales      | float | Product sales (thousands of units) — **target** |

## `survey_lung_cancer.csv`

Lung cancer survey dataset (309 rows after deduplication). Used in notebook **02** for binary classification.

| Column                 | Type    | Description                                |
|------------------------|---------|--------------------------------------------|
| GENDER                 | string  | `MALE` / `FEMALE`                          |
| AGE                    | int     | Age in years                               |
| SMOKING                | int 1/2 | Self-reported smoking                      |
| YELLOW_FINGERS         | int 1/2 | Yellow fingers symptom                     |
| ANXIETY                | int 1/2 | Anxiety symptom                            |
| PEER_PRESSURE          | int 1/2 | Peer pressure                              |
| CHRONIC DISEASE        | int 1/2 | Pre-existing chronic disease               |
| FATIGUE                | int 1/2 | Fatigue symptom                            |
| ALLERGY                | int 1/2 | Allergy                                    |
| WHEEZING               | int 1/2 | Wheezing                                   |
| ALCOHOL CONSUMING      | int 1/2 | Alcohol consumption                        |
| COUGHING               | int 1/2 | Coughing                                   |
| SHORTNESS OF BREATH    | int 1/2 | Shortness of breath                        |
| SWALLOWING DIFFICULTY  | int 1/2 | Swallowing difficulty                      |
| CHEST PAIN             | int 1/2 | Chest pain                                 |
| LUNG_CANCER            | string  | `YES` / `NO` — **target**                  |

> The 1/2 encoding is recoded to 0/1 inside the notebook for clarity.

## `telco_churn.csv` (generated)

Notebook **03** generates a synthetic Telco customer churn dataset (10k rows) in the notebook itself, mirroring the structure of the well-known IBM Telco Customer Churn dataset. The dataset is regenerated deterministically (`random_state=42`) so the notebook is fully reproducible without external downloads.
