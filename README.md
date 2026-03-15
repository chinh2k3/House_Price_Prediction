# House Price Prediction — Regression ML Pipeline

> Predict residential property sale prices using a fully reproducible, sklearn-compatible ML pipeline built on the Ames Housing dataset.

---

## Results

| Model | R² (test) | MAE | RMSE | RMSLE |
|---|---|---|---|---|
| Linear Regression | -7.26 | 37,274 | 251,643 | 0.7541 |
| **Ridge (best)** | **0.9167** | **14,519** | **25,278** | **0.1303** |
| Lasso (best) | 0.9126 | 14,619 | 25,896 | 0.1331 |

Best model: **Ridge Regression** — tuned via GridSearchCV (10-fold CV), trained on log-transformed target.

---

## Project Structure

```
├── artifacts/
│   ├── model.pkl
│   ├── preprocessor.pkl
│   └── scaler.pkl
├── Data/
│   ├── raw/
│   │   ├── train.csv
│   │   └── test.csv
│   ├── processed/
│   │   └── preprocessed.csv
│   └── prediction/
│       └── submission.csv
├── Notebooks/
│   ├── Preprocessing.ipynb
│   └── Modeling.ipynb
├── src/
│   ├── processing.py     # Custom Preprocessing class
│   ├── scaler.py         # SelectiveScaler (sklearn-compatible)
│   ├── model.py          # GridSearchCV Ridge training
│   ├── train.py          # End-to-end training script
│   └── predict.py        # Inference on test set
├── test/
│   └── test_preprocessing.py
└── README.md
```

---

## Pipeline Overview

### 1. Preprocessing (`processing.py`)

A custom sklearn-compatible `Preprocessing` class handles the full transformation:

**Missing values**
- Columns with >50% missing are dropped
- Missing-value indicator flags created before imputation
- Numerical: neighborhood-grouped median imputation (fallback to global median)
- Categorical: mode imputation

**Numeric feature categorization**
- `year/time` — clipped at 1800 as logical floor
- `zero_inflated` — winsorized on non-zero part + binary existence flag added
- `continuous` — winsorized (1st–99th percentile) + log-transformed if skew > 1
- `count` / `ordinal` — left unchanged

**Encoding**
- Ordinal features mapped to ordered integers (e.g. `ExterQual`: Po=1 → Ex=5)
- Nominal features one-hot encoded with `drop_first=True`

**Feature engineering**
- `HouseAge`, `YearsSinceRemod`, `Remodeled`
- `TotalArea`, `LogGrLivArea`, `LogTotalArea`, `LogLotArea`
- `QualArea` (OverallQual × GrLivArea)

---

### 2. Scaling (`scaler.py`)

`SelectiveScaler` extends `BaseEstimator` and `TransformerMixin`:
- Applies `StandardScaler` only to continuous numeric columns
- Skips binary flag columns (`_was_missing`, `_nonzero`) and one-hot dummies

---

### 3. Modeling (`model.py`)

- Target: `log1p(SalePrice)` — normalized right-skewed distribution
- Algorithm: **Ridge Regression**
- Tuning: `GridSearchCV` over 30 alpha values (log-spaced 1e-3 to 1e3), 10-fold CV
- Predictions: `expm1(clip(y_pred, 0, 15))` to convert back to price scale

---

## How to Run

### Install dependencies

```bash
pip install -r requirements.txt
```

### Train

```bash
python src/train.py
```

Saves `preprocessor.pkl`, `scaler.pkl`, and `model.pkl` to `artifacts/`.

### Predict

```bash
python src/predict.py
```

Reads `Data/raw/test.csv`, outputs `Data/prediction/submission.csv`.

---

## Dataset

- **Source:** [Kaggle — House Prices: Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques)
- **Size:** 1,460 rows × 81 columns (training set)
- **Target:** `SalePrice` (continuous, USD)

---

## Tech Stack

| Category | Libraries |
|---|---|
| Data processing | pandas, NumPy |
| ML & tuning | scikit-learn |
| Visualization | Matplotlib, Seaborn |
| Serialization | joblib |

---

## Key Design Decisions

- **No leakage:** scaler fitted only on training data; `preprocessor.transform()` is stateless after `fit()`
- **Fragmentation-free:** new derived columns batched via `pd.concat` instead of iterative insert
- **Reusable artifacts:** all fitted transformers serialized with `joblib` for production inference
