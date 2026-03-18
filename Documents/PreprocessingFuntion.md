# Preprocessing Functions

Tài liệu giải thích chi tiết từng method trong class `Preprocessing` (`src/processing.py`).

---

## Class: `Preprocessing`

```python
class Preprocessing:
    def __init__(self, drop_cols=None)
    def fit(self, df) -> self
    def transform(self, df) -> pd.DataFrame
    def fit_transform(self, df) -> pd.DataFrame
```

Tuân thủ sklearn `fit/transform` contract — có thể drop-in vào bất kỳ `sklearn.Pipeline` nào.

**Thuộc tính lưu state sau khi `fit()`:**

| Thuộc tính | Kiểu | Nội dung |
|---|---|---|
| `drop_miss_cols_` | list | Cột bị drop vì >50% missing |
| `miss_flag_cols_` | list | Cột có NaN lúc fit (cần tạo flag) |
| `num_medians_` | dict | `col → {neighborhood: median}` hoặc `col → scalar` |
| `cat_modes_` | dict | `col → mode value` |
| `continuous_cols_` | list | Cột continuous |
| `zero_inflated_` | list | Cột zero-inflated |
| `year_cols_` | list | Cột năm/thời gian |
| `winsor_bounds_` | dict | `col → (lower, upper)` |
| `log_cols_` | set | Continuous cols cần log transform |
| `nonzero_winsor_` | dict | `col → (lower, upper)` cho phần khác 0 |
| `nominal_cols_fit_` | list | Nominal cols thực sự có trong data lúc fit |
| `dummy_cols_` | Index | Schema columns sau one-hot encoding |

---

## `__init__(drop_cols=None)`

```python
pp = Preprocessing(drop_cols=["Id"])
```

**Tham số:**
- `drop_cols` *(list, optional)*: Danh sách cột cần drop trước khi xử lý. Thường dùng để bỏ ID columns không mang thông tin.

**Lưu ý:** Chỉ khởi tạo state rỗng — chưa học gì từ data. Phải gọi `fit()` trước khi `transform()`.

---

## `fit(df)`

Học toàn bộ thống kê từ training data. **Chỉ gọi trên train set.**

```python
pp = Preprocessing(drop_cols=["Id"])
pp.fit(X_train)
```

**Các bước thực hiện:**

**Bước 1 — Drop user-specified columns**
```python
df = df.drop(columns=self.drop_cols, errors="ignore")
```

**Bước 2 — Drop high-missing columns**
```python
miss_pct = df.isnull().mean()
self.drop_miss_cols_ = miss_pct[miss_pct > 0.5].index.tolist()
```
Lưu danh sách vào `drop_miss_cols_` để áp dụng lại trong `transform()`.

**Bước 3 — Ghi nhận cột có missing**
```python
self.miss_flag_cols_ = [c for c in df.columns if df[c].isnull().any()]
```
Dùng để tạo `_was_missing` flags trong `transform()`.

**Bước 4 — Fit imputation statistics**
- Numeric: median theo `Neighborhood` → `num_medians_[col] = {neighborhood: median_value}`
- Categorical: mode → `cat_modes_[col] = mode_value`

**Bước 5 — Impute (nội bộ, để fit outlier stats trên data sạch)**
Gọi `_impute()` trên bản copy của df trước khi fit winsorize bounds.

**Bước 6 — Phân loại numeric columns**
Gọi `_classify_numeric_cols()` → phân vào `continuous_cols_`, `zero_inflated_`, `year_cols_`.

**Bước 7 — Fit outlier treatment**
- Continuous: tính winsor bounds (1–99%), kiểm tra skewness → quyết định có log không
- Zero-inflated: tính winsor bounds chỉ trên phần khác 0

**Bước 8 — Fit one-hot encoding schema**
```python
df = pd.get_dummies(df, columns=self.nominal_cols_fit_, drop_first=True, dtype=int)
self.dummy_cols_ = df.columns   # lưu lại toàn bộ schema
```

**Returns:** `self` (để chain được: `pp.fit(X_train).transform(X_test)`)

---

## `transform(df)`

Áp dụng toàn bộ transformations đã học từ `fit()` lên data mới. **An toàn để gọi trên cả train lẫn test.**

```python
X_train_processed = pp.transform(X_train)
X_test_processed  = pp.transform(X_test)
```

**Các bước thực hiện:**

**Bước 1 — Drop columns**
```python
df = df.drop(columns=self.drop_cols, errors="ignore")
df = df.drop(columns=self.drop_miss_cols_, errors="ignore")
```

**Bước 2 — Tạo missing flags**
```python
for col in self.miss_flag_cols_:
    df[f"{col}_was_missing"] = df[col].isnull().astype(int)
```
Flag được tạo **trước** imputation để ghi nhận đúng vị trí NaN gốc.

**Bước 3 — Impute**
Gọi `_impute()` dùng stats đã fit: neighborhood-grouped median cho numeric, mode cho categorical.

**Bước 4a — Continuous: winsorize + log**
```python
df[col] = df[col].clip(lo, hi)          # winsorize
df[col + "_log"] = np.log1p(df[col])    # log nếu col in log_cols_
```

**Bước 4b — Zero-inflated: winsorize + nonzero flag**
```python
df.loc[~zero_mask, col] = df.loc[~zero_mask, col].clip(lo, hi)
df[col + "_nonzero"] = (~zero_mask).astype(int)
```
Flag `_nonzero` phân tách "có feature" (=1) khỏi "không có" (=0).

**Bước 4c — Year columns: clip**
```python
df[col] = df[col].clip(lower=1800)
```
Giới hạn hạ tại 1800 để loại bỏ giá trị vô lý (năm = 0, năm âm).

**Bước 5 — Ordinal encoding**
```python
df[col] = df[col].map(mapping).fillna(-1)
```
Category không có trong mapping (unseen) → -1, không gây lỗi.

**Bước 6 — Feature engineering**
Gọi `feature_engineering()` static method.

**Bước 7 — One-hot encoding + align schema**
```python
df = pd.get_dummies(df, columns=nominal_present, drop_first=True, dtype=int)
df = df.reindex(columns=self.dummy_cols_, fill_value=0)
```
`reindex` đảm bảo test set có đúng columns như train — cột thiếu được fill = 0, cột thừa bị drop.

**Returns:** `pd.DataFrame` với shape `(n_rows, len(dummy_cols_))`

---

## `fit_transform(df)`

Shorthand gọi `fit(df).transform(df)`.

```python
X_train_processed = pp.fit_transform(X_train)
```

**Quan trọng:** Chỉ dùng trên **train set**. Không dùng trên test set vì sẽ fit lại stats từ test data — gây data leakage.

---

## `_impute(df, num_cols, cat_cols)` *(internal)*

Method nội bộ, dùng chung giữa `fit()` và `transform()`.

```python
df = self._impute(df, num_cols, cat_cols)
```

**Logic numeric imputation:**
```python
# Ưu tiên neighborhood-grouped median
mapped = df["Neighborhood"].map(self.num_medians_[col])
df[col] = df[col].fillna(mapped)
# Fallback: global median nếu neighborhood cũng thiếu
df[col] = df[col].fillna(np.median(list(self.num_medians_[col].values())))
```

**Logic categorical imputation:**
```python
df[col] = df[col].fillna(self.cat_modes_.get(col, "Missing"))
```

---

## `_classify_numeric_cols(df, num_cols)` *(internal)*

Phân loại numeric columns thành 3 nhóm chính.

```python
continuous, zero_inflated, year_cols = self._classify_numeric_cols(df, num_cols)
```

**Logic phân loại:**

```python
# Year cols: tên chứa "Year" hoặc "Yr"
year_cols = [c for c in num_cols if "Year" in c or "Yr" in c]

# Zero-inflated: >30% giá trị = 0 VÀ Q25 = 0
zero_inflated = [c for c in num_cols
                 if zero_ratio[c] > 0.3
                 and df[c].quantile(0.25) == 0]

# Count/ordinal: nunique < 10
count_cols = [c for c in num_cols if nunique[c] < 10]

# Continuous: còn lại (không thuộc 3 nhóm trên, không phải ordinal)
continuous = [c for c in num_cols
              if c not in year_cols
              and c not in zero_inflated
              and c not in count_cols
              and c not in ORDINAL_MAPPINGS]
```

`_was_missing` flag cols được loại khỏi tất cả các nhóm (chỉ là binary, không cần xử lý thêm).

---

## `feature_engineering(df)` *(static method)*

```python
df = Preprocessing.feature_engineering(df)
```

Có thể gọi độc lập mà không cần instance. Tạo 8 features mới:

```python
df["HouseAge"]         = df["YrSold"] - df["YearBuilt"]
df["YearsSinceRemod"]  = df["YrSold"] - df["YearRemodAdd"]
df["Remodeled"]        = (df["YearBuilt"] != df["YearRemodAdd"]).astype(int)
df["HasLotFrontage"]   = (~df["LotFrontage"].isna()).astype(int)
df["TotalArea"]        = df["GrLivArea"] + df["TotalBsmtSF"] + df["GarageArea"] \
                       + df["1stFlrSF"] + df["2ndFlrSF"]
df["LogGrLivArea"]     = np.log1p(df["GrLivArea"])
df["LogTotalArea"]     = np.log1p(df["TotalArea"])
df["LogLotArea"]       = np.log1p(df["LotArea"])
df["QualArea"]         = df["OverallQual"] * df["GrLivArea"]
```

**Lưu ý:** Method này được gọi trong cả `fit()` lẫn `transform()` — đảm bảo các feature mới luôn được tạo nhất quán.

---

## `_winsorize(series, lower_q=0.01, upper_q=0.99)` *(static method)*

```python
series_clipped, lo, hi = Preprocessing._winsorize(series)
```

Clip giá trị nằm ngoài khoảng [1st percentile, 99th percentile] về đúng giới hạn đó. Giảm ảnh hưởng của outliers mà không loại bỏ hoàn toàn các điểm dữ liệu.

**Returns:** `(clipped_series, lower_bound, upper_bound)`

---

## Class constants

### `ORDINAL_MAPPINGS`

Dictionary ánh xạ 17 cột ordinal sang số nguyên theo thứ tự chất lượng tăng dần:

```python
"ExterQual": {"Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5}
"Functional": {"Sev": 1, "Maj2": 2, "Maj1": 3, "Mod": 4, "Min2": 5, "Min1": 6, "Typ": 7}
# ... 15 cột khác
```

### `NOMINAL_COLS`

21 cột categorical không có thứ tự sẽ được one-hot encoded:
`MSZoning`, `Street`, `Neighborhood`, `BldgType`, `HouseStyle`, `SaleType`, v.v.

### `SPECIAL_NO_LOG`

```python
SPECIAL_NO_LOG = ["TotalBsmtSF"]
```

Các cột continuous nhưng **không** áp dụng log transform dù có skewness cao — vì `TotalBsmtSF = 0` có ý nghĩa thực (không có tầng hầm), log transform sẽ bị xử lý riêng qua zero-inflated pipeline thay thế.