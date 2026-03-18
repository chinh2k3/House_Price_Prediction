# Pipeline Overview

Tài liệu này mô tả toàn bộ flow từ raw data đến prediction trong dự án House Price Prediction.

## 1. Data Loading (`train.py`)

**Input:** `Data/raw/train.csv` — 1,460 rows × 81 columns

```python
X = df.drop(columns=["SalePrice"])
y = np.log1p(df["SalePrice"])          # log-transform target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

**Lý do log-transform target:**
SalePrice có skewness ≈ 1.88 (right-skewed). Log transform đưa về gần phân phối chuẩn (skew ≈ 0.12), phù hợp với giả định của Ridge Regression. Khi predict sẽ dùng `expm1()` để chuyển ngược lại.

---

## 2. Preprocessing (`processing.py`)

### 2.1 Drop high-missing columns
Các cột có >50% giá trị bị thiếu bị loại bỏ hoàn toàn (ví dụ: `PoolQC`, `Alley`, `Fence`). Ngưỡng 50% được chọn vì imputation trên dữ liệu quá thưa sẽ không đáng tin cậy.

### 2.2 Missing value flags
Trước khi impute, tạo cột `{col}_was_missing` = 1 nếu giá trị gốc bị thiếu, 0 nếu không. Điều này giữ lại thông tin rằng dữ liệu đã từng bị thiếu — đôi khi bản thân sự thiếu hụt mang ý nghĩa (ví dụ `GarageType = NaN` nghĩa là không có garage).

### 2.3 Imputation
- **Numeric:** median theo từng `Neighborhood` — nhà cùng khu vực có giá trị tương tự nhau, chính xác hơn global median
- **Categorical:** mode của toàn cột

### 2.4 Phân loại numeric columns

| Loại | Điều kiện | Xử lý |
|---|---|---|
| `continuous` | Không thuộc các loại khác | Winsorize (1–99%) + log nếu skew > 1 |
| `zero_inflated` | >30% giá trị = 0, Q25 = 0 | Winsorize phần khác 0 + thêm `_nonzero` flag |
| `year/time` | Tên chứa "Year" hoặc "Yr" | Clip tại 1800 |
| `count/ordinal` | nunique < 10 | Giữ nguyên |

### 2.5 Ordinal encoding
Các cột thứ tự được map sang số nguyên dựa trên domain knowledge:

```
ExterQual: Po=1, Fa=2, TA=3, Gd=4, Ex=5
Functional: Sev=1, Maj2=2, Maj1=3, Mod=4, Min2=5, Min1=6, Typ=7
```

Category không có trong mapping → giá trị -1.

### 2.6 Feature Engineering

| Feature mới | Công thức | Ý nghĩa |
|---|---|---|
| `HouseAge` | `YrSold - YearBuilt` | Tuổi nhà |
| `YearsSinceRemod` | `YrSold - YearRemodAdd` | Số năm kể từ lần sửa chữa cuối |
| `Remodeled` | `YearBuilt != YearRemodAdd` | Nhà đã từng được cải tạo chưa |
| `TotalArea` | `GrLivArea + TotalBsmtSF + GarageArea + 1stFlrSF + 2ndFlrSF` | Tổng diện tích |
| `LogGrLivArea` | `log1p(GrLivArea)` | Log diện tích sinh hoạt |
| `LogTotalArea` | `log1p(TotalArea)` | Log tổng diện tích |
| `LogLotArea` | `log1p(LotArea)` | Log diện tích lô đất |
| `QualArea` | `OverallQual × GrLivArea` | Interaction: chất lượng × diện tích |

### 2.7 One-hot encoding
21 cột nominal được one-hot encode với `drop_first=True` để tránh multicollinearity. Schema columns được lưu lại trong `dummy_cols_` — khi transform test set, dùng `reindex` để đảm bảo columns khớp hoàn toàn với train.

---

## 3. Scaling (`scaler.py`)

`SelectiveScaler` chỉ apply `StandardScaler` lên continuous numeric columns, bỏ qua:
- `_was_missing` flags (binary 0/1)
- `_nonzero` flags (binary 0/1)
- One-hot dummy columns (binary 0/1)

Scale các cột binary làm mất ý nghĩa và gây nhiễu cho model.

> **Lưu ý:** XGBoost là tree-based model, không bị ảnh hưởng bởi scale. Trong notebook, XGBoost được train và evaluate trên `X_train_pre` (chưa scale) — tách biệt với `X_train` đã scale dùng cho Ridge/Lasso.

---

## 4. Modeling (`model.py`)

**Model được chọn:** Ridge Regression sau khi so sánh với Lasso, ElasticNet và XGBoost.

```
GridSearchCV:
  - alphas: 30 giá trị log-spaced từ 1e-3 đến 1e3
  - cv: KFold(n_splits=5, shuffle=True, random_state=42)
  - scoring: R²
  - Best alpha: 2.04
  - Best CV R²: 0.8878
```

**Kết quả final trên test set:**

| Model | Test R² | RMSLE |
|---|---|---|
| Linear Regression | -7.26 | 0.7541 |
| XGBoost (tuned) | 0.9036 | 0.1391 |
| Lasso (tuned) | 0.9126 | 0.1331 |
| **Ridge (tuned)** | **0.9167** | **0.1303** |

---

## 5. Artifacts & Inference (`predict.py`)

Ba artifacts được lưu bằng `joblib`:

| File | Nội dung |
|---|---|
| `preprocessor.pkl` | Fitted `Preprocessing` object — chứa medians, modes, winsor bounds, dummy schema |
| `scaler.pkl` | Fitted `SelectiveScaler` — chứa StandardScaler params |
| `model.pkl` | Fitted `Ridge` best estimator |

Khi predict:
```python
X_test = preprocessor.transform(df_test)   # dùng stats đã fit từ train
X_test = scaler.transform(X_test)
y_log  = model.predict(X_test)
y_pred = np.expm1(np.clip(y_log, 0, 15))  # clip để tránh overflow
```

`clip(0, 15)` đảm bảo không có giá trị log âm (nhà giá âm) hoặc quá lớn (e^15 ≈ 3.3 triệu USD).