# House Price Prediction — Regression ML Pipeline

> Predict residential property sale prices using a fully reproducible, sklearn-compatible ML pipeline built on the Ames Housing dataset.  
> Focus: end-to-end pipeline design, leakage prevention, and interpretable regularization.

---

## Results

| Model | R² (test) | MAE | RMSE | RMSLE |
|---|---|---|---|---|
| Naive baseline (predict median) | ~0.00 | ~55,000 | ~79,000 | — |
| Linear Regression | -7.26 | 37,274 | 251,643 | 0.7541 |
| XGBoost (best) | 0.9036 | 14,936 | 27,194 | 0.1391 |
| Lasso (best) | 0.9126 | 14,619 | 25,896 | 0.1331 |
| **Ridge (best)** | **0.9167** | **14,519** | **25,278** | **0.1303** |

> Best model: **Ridge Regression** tuned via GridSearchCV (5-fold CV), trained on log-transformed target.  
> Ridge outperformed all models including tree-based XGBoost — a result explained by the dataset characteristics: 1,168 training samples with 234 features creates a high-dimensional, multicollinear space where L2 regularization excels.

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
│   ├── Preprocessing.ipynb       # EDA, outlier analysis, encoding decisions
│   └── Modeling.ipynb            # Baseline → regularization → XGBoost → tuning → evaluation
├── src/
│   ├── processing.py             # Custom Preprocessing class (fit/transform)
│   ├── scaler.py                 # SelectiveScaler (sklearn-compatible)
│   ├── model.py                  # GridSearchCV Ridge training
│   ├── train.py                  # End-to-end training script
│   └── predict.py                # Inference on test set
├── test/
│   └── test_preprocessing.py
└── README.md
```

---

## Pipeline Overview

### 1. Preprocessing (`processing.py`)

**Missing values**
- Columns with >50% missing are dropped (e.g. `PoolQC`, `Alley`, `Fence`)
- Missing-value indicator flags (`_was_missing`) are created *before* imputation to preserve the signal that data was absent
- Numerical: neighborhood-grouped median imputation — houses in the same neighborhood share similar price ranges, making this more accurate than a global median
- Categorical: mode imputation

**Numeric feature categorization**

Rather than applying a single strategy to all numeric columns, features were split into 5 types to allow tailored treatment:

| Type | Strategy | Rationale |
|---|---|---|
| `continuous` | Winsorize (1–99%) + log if skew > 1 | Reduce outlier influence; normalize distribution for Ridge |
| `zero_inflated` | Winsorize non-zero part + add `_nonzero` flag | Separates "has feature" from "how much" |
| `year/time` | Clip at 1800 | Logical floor; no statistical outlier treatment needed |
| `count` / `ordinal` | Unchanged | Low cardinality; statistical treatment would distort meaning |

**Encoding**
- Ordinal features mapped to ordered integers based on domain knowledge (e.g. `ExterQual`: Po=1 → Ex=5)
- Nominal features one-hot encoded with `drop_first=True` to avoid multicollinearity

**Feature engineering**
- `HouseAge`, `YearsSinceRemod`, `Remodeled` — captures depreciation and renovation signal
- `TotalArea` — combines living, basement, garage, and floor areas
- `LogGrLivArea`, `LogTotalArea`, `LogLotArea` — pre-log versions for direct use in model
- `QualArea` (OverallQual × GrLivArea) — interaction term: quality multiplied by size

---

### 2. Scaling (`scaler.py`)

`SelectiveScaler` extends `BaseEstimator` and `TransformerMixin`:
- Applies `StandardScaler` only to continuous numeric columns
- Deliberately skips binary flag columns (`_was_missing`, `_nonzero`) and one-hot dummies — scaling binary features distorts their meaning and hurts model interpretability
- XGBoost was evaluated on unscaled data (`X_train_pre`) to respect its tree-based nature

---

### 3. Modeling (`model.py`)

- **Target transformation:** `log1p(SalePrice)` — SalePrice is right-skewed (skew ≈ 1.88); log transform brings it close to normal, which Ridge assumes
- **Algorithm:** Ridge Regression
- **Tuning:** GridSearchCV over 30 alpha values (log-spaced 1e-3 to 1e3), 5-fold CV, scoring on R²
- **Prediction:** `expm1(clip(y_pred, 0, 15))` to safely convert back to price scale

---

## Model Selection

Four models were evaluated using 5-fold cross-validation and held-out test set metrics:

| Model | CV R² (mean ± std) | Test R² | Test RMSLE |
|---|---|---|---|
| Ridge (α=1.0, default) | 0.8866 ± 0.0082 | — | — |
| Lasso (α=0.001, default) | 0.8881 ± 0.0241 | — | — |
| ElasticNet (α=0.001) | 0.8852 ± 0.0226 | — | — |
| XGBoost (default) | 0.8964 ± 0.0173 | — | — |
| Ridge (best α=2.04) | 0.8878 | **0.9167** | **0.1303** |
| Lasso (best α=0.0035) | 0.8929 | 0.9126 | 0.1331 |
| XGBoost (tuned) | **0.9069** | 0.9036 | 0.1391 |

**Why Ridge wins over XGBoost:** Despite XGBoost achieving the highest CV R² (0.9069), Ridge generalizes better on the test set (0.9167 vs 0.9036). With only 1,168 training samples and 234 features, the dataset is too small for tree-based models to fully exploit their capacity — XGBoost showed train R² = 0.9939 vs test R² = 0.9036, indicating residual overfitting even after full tuning (`max_depth=3`, `gamma`, `min_child_weight`). Ridge's L2 penalty is better suited to this high-dimensional, multicollinear setting.

---

## Challenges & Solutions

**1. Severe overfitting with Linear Regression**

Plain Linear Regression achieved R² = 0.936 on train but collapsed to R² = -7.26 on test — a sign of severe multicollinearity across 200+ features after one-hot encoding. Ridge regularization penalizes large coefficients and reduced the test error to R² = 0.917, closing the train-test gap from 8.2 to under 0.03.

**2. Data leakage risk in preprocessing**

Fitting imputation statistics (medians, modes) or scaler parameters on the full dataset before splitting would leak test information into training. All statistics are fitted exclusively on training data inside `Preprocessing.fit()`, then applied via `transform()` on test data — the same contract sklearn pipelines enforce.

**3. DataFrame fragmentation warning**

Iteratively inserting new columns (`df[new_col] = data`) inside loops caused pandas `PerformanceWarning` due to internal memory fragmentation. Fixed by collecting all new columns into a dictionary first, then joining in a single `pd.concat` call — eliminating the warning and improving transform speed.

**4. Heterogeneous numeric features requiring different treatment**

Applying a single outlier strategy to all numeric columns distorts meaning: clipping `YearBuilt` at the 99th percentile makes no sense, and `GarageArea = 0` is a valid value (no garage), not an outlier. Solution: classify numeric columns into 5 types and apply tailored strategies per type.

**5. XGBoost data consistency**

XGBoost is scale-invariant as a tree-based model and must be evaluated on unscaled data. A separate `X_train_pre` / `X_test_pre` variable (pre-StandardScaler) was maintained throughout the notebook to ensure consistent input across fit, cross-validation, GridSearch, and final evaluation.

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

Fits preprocessor and scaler on training data, runs GridSearchCV, saves `preprocessor.pkl`, `scaler.pkl`, and `model.pkl` to `artifacts/`.

### Predict

```bash
python src/predict.py
```

Loads fitted artifacts, transforms `Data/raw/test.csv`, outputs predictions to `Data/prediction/submission.csv`.

---

## Dataset

- **Source:** [Kaggle — House Prices: Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques)
- **Size:** 1,460 rows × 81 columns (training set)
- **Target:** `SalePrice` (continuous, USD)
- **Notable characteristics:** 36 numeric + 43 categorical features; heavy missingness in several columns; strong multicollinearity between area-related features

---

## Tech Stack

| Category | Libraries |
|---|---|
| Data processing | pandas, NumPy |
| ML & tuning | scikit-learn, XGBoost |
| Visualization | Matplotlib, Seaborn |
| Serialization | joblib |

---

## Key Design Decisions

- **No leakage:** all fit statistics derived exclusively from training data; `transform()` is purely stateless
- **Sklearn-compatible classes:** `Preprocessing` and `SelectiveScaler` follow the `fit/transform` contract, making them droppable into any sklearn `Pipeline`
- **Fragmentation-free:** new derived columns batched via `pd.concat` instead of iterative insert
- **Reusable artifacts:** all fitted transformers serialized with `joblib` for consistent inference on new data
- **Model selection with evidence:** Ridge chosen over XGBoost after full GridSearch tuning of both — not assumed, verified