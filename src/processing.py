import pandas as pd
import numpy as np

class Preprocessing:
    #  Column type definitions
    ORDINAL_MAPPINGS = {
        "LotShape":    {"Reg": 4, "IR1": 3, "IR2": 2, "IR3": 1},
        "LandSlope":   {"Gtl": 3, "Mod": 2, "Sev": 1},
        "ExterQual":   {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1},
        "ExterCond":   {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1},
        "BsmtQual":    {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1},
        "BsmtCond":    {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1},
        "HeatingQC":   {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1},
        "KitchenQual": {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1},
        "FireplaceQu": {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1},
        "GarageQual":  {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1},
        "GarageCond":  {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1},
        "BsmtExposure":  {"Gd": 4, "Av": 3, "Mn": 2, "No": 1},
        "BsmtFinType1":  {"GLQ": 6, "ALQ": 5, "BLQ": 4, "Rec": 3, "LwQ": 2, "Unf": 1},
        "BsmtFinType2":  {"GLQ": 6, "ALQ": 5, "BLQ": 4, "Rec": 3, "LwQ": 2, "Unf": 1},
        "Functional":    {"Typ": 7, "Min1": 6, "Min2": 5, "Mod": 4,
                          "Maj1": 3, "Maj2": 2, "Sev": 1},
        "Electrical":    {"SBrkr": 5, "FuseA": 4, "FuseF": 3, "FuseP": 2, "Mix": 1},
        "PavedDrive":    {"Y": 3, "P": 2, "N": 1},
    }

    NOMINAL_COLS = [
        "MSZoning", "Street", "LandContour", "Utilities", "LotConfig",
        "Neighborhood", "Condition1", "Condition2", "BldgType", "HouseStyle",
        "RoofStyle", "RoofMatl", "Exterior1st", "Exterior2nd", "Foundation",
        "Heating", "CentralAir", "GarageType", "GarageFinish",
        "SaleType", "SaleCondition",
    ]

    # Columns where log transform is skipped
    SPECIAL_NO_LOG = ["TotalBsmtSF"]

    def __init__(self, drop_cols = None):
        self.drop_cols = drop_cols if drop_cols else []

        # Fitted state
        self.drop_miss_cols_   = []
        self.miss_flag_cols_   = [] # columns that had missing values at fit-time
        self.num_medians_      = {} # col → median or {neighborhood: median}
        self.cat_modes_        = {} # col → mode value

        self.continuous_cols_  = []
        self.zero_inflated_    = []
        self.year_cols_        = []

        self.winsor_bounds_    = {} # col → (lower, upper)  for continuous
        self.log_cols_         = set() # continuous cols that get log-transformed
        self.nonzero_winsor_   = {} # col → (lower, upper)  for zero-inflated non-zero part

        self.nominal_cols_fit_ = [] # nominal cols actually present at fit time
        self.dummy_cols_       = None # column index after get_dummies

    # Statics Method
    @staticmethod
    def _winsorize(series, lower_q = 0.01, upper_q = 0.99):
        lo = series.quantile(lower_q)
        hi = series.quantile(upper_q)
        return series.clip(lo,hi), lo, hi

    # Split numeric columns into continuous / zero_inflated / year / (ordinal+count)
    def _classify_numeric_cols(self, df, num_cols):
        flag_cols = [c for c in num_cols if c.endswith("_was_missing")]
        nunique = df[num_cols].nunique()
        zero_ratio = (df[num_cols] == 0).mean()

        year_cols = [c for c in num_cols if ("Year" in c or "Yr" in c) and c not in flag_cols]

        zero_inflated = [c for c in num_cols if zero_ratio[c] > 0.3 and df[c].quantile(0.25) == 0 and c not in flag_cols and c not in year_cols]

        count_cols = [c for c in num_cols if nunique[c] < 10 and c not in flag_cols and c not in zero_inflated and c not in year_cols]
        
        continuous = [c for c in num_cols if c not in flag_cols and c not in zero_inflated and c not in year_cols and c not in count_cols and c not in list(self.ORDINAL_MAPPINGS.keys())]

        return continuous, zero_inflated, year_cols

    # Feature Engineering
    @staticmethod
    def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["HouseAge"] = df["YrSold"] - df["YearBuilt"]
        df["YearsSinceRemod"] = df["YrSold"] - df["YearRemodAdd"]
        df["Remodeled"] = (df["YearBuilt"] != df["YearRemodAdd"]).astype(int)
        df["HasLotFrontage"] = (~df["LotFrontage"].isna()).astype(int)
        df["TotalArea"] = (df["GrLivArea"] + df["TotalBsmtSF"] + df["GarageArea"] + df["1stFlrSF"] + df["2ndFlrSF"])
        df["LogGrLivArea"] = np.log1p(df["GrLivArea"])
        df["LogTotalArea"] = np.log1p(df["TotalArea"])
        df["LogLotArea"] = np.log1p(df["LotArea"])
        df["QualArea"] = df["OverallQual"] * df["GrLivArea"]
        return df

    # Fit
    def fit(self, df):
        df = df.copy()

        # 1. Drop user-specified columns
        df = df.drop(columns=self.drop_cols, errors='ignore')

        # 2. Drop columns with > 50 % missing
        miss_pct = df.isnull().mean()
        self.drop_miss_cols_ = miss_pct[miss_pct > 0.5].index.tolist()
        df = df.drop(columns=self.drop_miss_cols_)

        # 3. Record which columns had missing values
        self.miss_flag_cols_ = [c for c in df.columns if df[c].isnull().any()]
        
        # 4. Identify column types
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        # 5. Fit imputation statistics
        if "Neighborhood" in df.columns:
            for col in num_cols:
                self.num_medians_[col] = df.groupby("Neighborhood")[col].median().to_dict()
        else:
            for col in num_cols:
                self.num_medians_[col] = df[col].median()

        for col in cat_cols:
            mode = df[col].mode()
            self.cat_modes_[col] = mode.iloc[0] if len(mode) > 0 else 'Missing'

        # 6. Apply imputation to df for fitting outlier/encoding stats
        df = self._impute(df, num_cols, cat_cols)

        # 7. Classify numeric columns
        self.continuous_cols_, self.zero_inflated_, self.year_cols_ = (self._classify_numeric_cols(df, num_cols))

        # 8. Fit outlier treatment
        # 8a. Continuous → winsorize + optional log
        for col in self.continuous_cols_:
            series = df[col].astype(float)
            skew_before = series.skew()
            series_win, lo, hi = self._winsorize(series) if abs(skew_before) > 2 else (series, series.quantile(0.01), series.quantile(0.99))
        
            # After Winsorize
            _, lo, hi = self._winsorize(df[col].astype(float))
            self.winsor_bounds_[col] = (lo,hi)

            # Log_transform
            skew_mid = series_win.skew()
            if col not in self.SPECIAL_NO_LOG and abs(skew_mid) > 1:
                self.log_cols_.add(col)

        # 8b. Zero-inflated-> winsorize non-zero part
        for col in self.zero_inflated_:
            non_zero = df.loc[df[col] != 0, col].astype(float)
            if len(non_zero) > 0:
                lo = non_zero.quantile(0.01)
                hi = non_zero.quantile(0.99)
                self.nonzero_winsor_[col] = (lo,hi)

        # 9. Encode ordinal cols
        for col, mapping in self.ORDINAL_MAPPINGS.items():
            if col in df.columns:
                df[col] = df[col].map(mapping).fillna(-1)

        # 10. Feature Engineering
        df = self.feature_engineering(df)

        # 11. One-hot Encoding - fit columns schema
        self.nominal_cols_fit_ = [c for c in self.NOMINAL_COLS if c in df.columns]
        df = pd.get_dummies(df, columns = self.nominal_cols_fit_, drop_first = True, dtype = int)
        self.dummy_cols_ = df.columns
        return self

    # Internal imputer (shared between fit and transform)
    def _impute(self, df, num_cols, cat_cols):
        # Numeric: neighborhood-grouped median, fallback to global median
        for col in num_cols:
            if col not in df.columns:
                continue
            if df[col].isnull().any():
                if "Neighborhood" in df.columns and isinstance(self.num_medians_.get(col), dict):
                    mapped = df["Neighborhood"].map(self.num_medians_[col])
                    df[col] = df[col].fillna(mapped)
                    df[col] = df[col].fillna(np.median(list(self.num_medians_[col].values())))
                else:
                    df[col] = df[col].fillna(self.num_medians_.get(col, df[col].median()))

        # Categorical: mode
        for col in cat_cols:
            if col in df.columns:
                df[col] = df[col].fillna(self.cat_modes_.get(col, "Missing"))
        return df
    
    # Transform
    def transform(self, df:pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # 1. Drop columns
        df = df.drop(columns = self.drop_cols, errors = "ignore")
        df = df.drop(columns = self.drop_miss_cols_, errors = "ignore")

        # 2. Missing flags
        flag_data = {
            f"{col}_was_missing": df[col].isnull().astype(int)
            for col in self.miss_flag_cols_ if col in df.columns
        }
        if flag_data:
            df = pd.concat([df, pd.DataFrame(flag_data, index=df.index)], axis=1)

        # 3. Impute
        cat_cols = df.select_dtypes(include = ["object","category"]).columns.tolist()
        num_cols = df.select_dtypes(include = [np.number]).columns.tolist()
        df = self._impute(df, num_cols, cat_cols)

        # 4. Outlier treatment
        # 4a. Continuous -> winsorize + log
        log_data = {}
        for col in self.continuous_cols_:
            if col not in df.columns:
                continue
            lo, hi = self.winsor_bounds_.get(col, (None, None))
            if lo is not None:
                df[col] = df[col].astype(float).clip(lo,hi)
            if col in self.log_cols_:
                log_data[col + "_log"] = np.log1p(df[col])
        if log_data:
            df = pd.concat([df, pd.DataFrame(log_data, index = df.index)], axis = 1)
        
        # 4b. Zero-inflated -> winsorize non-zero + _nonzero flag
        nonzero_data = {}
        for col in self.zero_inflated_:
            if col not in df.columns:
                continue
            df[col] = df[col].astype(float)
            zero_mask = df[col] == 0
            if col in self.nonzero_winsor_:
                lo, hi = self.nonzero_winsor_[col]
                df.loc[~zero_mask, col] = df.loc[~zero_mask, col].clip(lo, hi)
            nonzero_data[col + "_nonzero"] = (~zero_mask).astype(int)
        if nonzero_data:
            df = pd.concat([df, pd.DataFrame(nonzero_data, index = df.index)], axis = 1)
        # 4c. Year/Time -> clip
        for col in self.year_cols_:
            if col in df.columns:
                df[col] = df[col].clip(lower = 1800)

        # 5. Ordinal Encoding
        for col, mapping in self.ORDINAL_MAPPINGS.items():
            if col in df.columns:
                df[col] = df[col].map(mapping).fillna(-1)

        # 6. Feature Engineering
        df = self.feature_engineering(df)

        # 7. One-hot encoding + align columns
        nominal_present = [c for c in self.nominal_cols_fit_ if c in df.columns]
        df = pd.get_dummies(df, columns = nominal_present, drop_first = True, dtype = int)
        df = df.reindex(columns = self.dummy_cols_, fill_value = 0)
        
        return df

    # Convenience
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)