import pytest
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from processing import Preprocessing

def make_df(n=100, seed=42):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "LotArea":      rng.integers(5000, 20000, n).astype(float),
        "GrLivArea":    rng.integers(800, 3000, n).astype(float),
        "TotalBsmtSF":  rng.integers(0, 2000, n).astype(float),
        "GarageArea":   rng.integers(0, 800, n).astype(float),
        "1stFlrSF":     rng.integers(500, 2000, n).astype(float),
        "2ndFlrSF":     rng.integers(0, 1000, n).astype(float),
        "MasVnrArea":   rng.integers(0, 500, n).astype(float),
        "YearBuilt":    rng.integers(1950, 2010, n).astype(float),
        "YearRemodAdd": rng.integers(1960, 2015, n).astype(float),
        "YrSold":       rng.integers(2006, 2011, n).astype(float),
        "OverallQual":  rng.integers(1, 10, n).astype(float),
        "OverallCond":  rng.integers(1, 9, n).astype(float),
        "WoodDeckSF":   np.where(rng.random(n) < 0.5, 0, rng.integers(100, 500, n)).astype(float),
        "ExterQual":    rng.choice(["Ex", "Gd", "TA", "Fa"], n),
        "KitchenQual":  rng.choice(["Ex", "Gd", "TA", "Fa"], n),
        "LotShape":     rng.choice(["Reg", "IR1", "IR2"], n),
        "PavedDrive":   rng.choice(["Y", "P", "N"], n),
        "HeatingQC":    rng.choice(["Ex", "Gd", "TA"], n),
        "BsmtQual":     rng.choice(["Ex", "Gd", "TA", None], n),
        "FireplaceQu":  rng.choice(["Gd", "TA", None], n),
        "GarageQual":   rng.choice(["Gd", "TA", "Fa", None], n),
        "LandSlope":    rng.choice(["Gtl", "Mod", "Sev"], n),
        "ExterCond":    rng.choice(["Ex", "Gd", "TA"], n),
        "Functional":   rng.choice(["Typ", "Min1", "Min2"], n),
        "Electrical":   rng.choice(["SBrkr", "FuseA", None], n),
        "BsmtCond":     rng.choice(["Gd", "TA", "Fa", None], n),
        "BsmtExposure": rng.choice(["Gd", "Av", "No", None], n),
        "BsmtFinType1": rng.choice(["GLQ", "ALQ", "Unf", None], n),
        "BsmtFinType2": rng.choice(["Rec", "LwQ", "Unf", None], n),
        "GarageCond":   rng.choice(["Gd", "TA", "Fa", None], n),
        "Neighborhood": rng.choice(["NAmes", "CollgCr", "OldTown", "Edwards"], n),
        "MSZoning":     rng.choice(["RL", "RM", "C (all)"], n),
        "Street":       rng.choice(["Pave", "Grvl"], n),
        "LandContour":  rng.choice(["Lvl", "Bnk", "HLS"], n),
        "Utilities":    rng.choice(["AllPub", "NoSeWa"], n),
        "LotConfig":    rng.choice(["Inside", "Corner", "CulDSac"], n),
        "Condition1":   rng.choice(["Norm", "Feedr", "PosN"], n),
        "Condition2":   rng.choice(["Norm", "Feedr"], n),
        "BldgType":     rng.choice(["1Fam", "2fmCon", "Duplex"], n),
        "HouseStyle":   rng.choice(["1Story", "2Story", "1.5Fin"], n),
        "RoofStyle":    rng.choice(["Gable", "Hip"], n),
        "RoofMatl":     rng.choice(["CompShg", "WdShngl"], n),
        "Exterior1st":  rng.choice(["VinylSd", "HdBoard", "MetalSd"], n),
        "Exterior2nd":  rng.choice(["VinylSd", "HdBoard", "Wd Shng"], n),
        "Foundation":   rng.choice(["PConc", "CBlock", "BrkTil"], n),
        "Heating":      rng.choice(["GasA", "GasW"], n),
        "CentralAir":   rng.choice(["Y", "N"], n),
        "GarageType":   rng.choice(["Attchd", "Detchd", None], n),
        "GarageFinish": rng.choice(["Fin", "RFn", "Unf", None], n),
        "SaleType":     rng.choice(["WD", "New", "COD"], n),
        "SaleCondition":rng.choice(["Normal", "Abnorml", "Partial"], n),
        "PoolQC":       [None] * n,   # >50% missing → should be dropped
    })


@pytest.fixture
def df_train():
    return make_df(n=200, seed=0)

@pytest.fixture
def df_test():
    return make_df(n=50, seed=99)

@pytest.fixture
def pp(df_train):
    p = Preprocessing(drop_cols=["Id"])
    p.fit(df_train)
    return p

# 1. Shape — row count preserved, train/test columns identical
def test_row_count_preserved(df_train):
    out = Preprocessing().fit_transform(df_train)
    assert out.shape[0] == len(df_train)

def test_train_test_columns_match(pp, df_train, df_test):
    assert list(pp.transform(df_train).columns) == list(pp.transform(df_test).columns)

# 2. No NaN after transform
def test_no_nan_train(df_train):
    out = Preprocessing().fit_transform(df_train)
    assert out.isnull().sum().sum() == 0

def test_no_nan_test(pp, df_test):
    out = pp.transform(df_test)
    assert out.isnull().sum().sum() == 0

# 3. No object columns remain
def test_no_object_columns(df_train):
    out = Preprocessing().fit_transform(df_train)
    assert out.select_dtypes(include="object").empty

# 4. High-missing column dropped
def test_poolqc_dropped(df_train):
    out = Preprocessing().fit_transform(df_train)
    assert "PoolQC" not in out.columns

# 5. Engineered features present
def test_engineered_features_exist(df_train):
    out = Preprocessing().fit_transform(df_train)
    for col in ["HouseAge", "YearsSinceRemod", "TotalArea", "QualArea"]:
        assert col in out.columns, f"Missing: {col}"

# 6. Transform is stateless (no side effects between calls)
def test_transform_stateless(pp, df_test):
    out1 = pp.transform(df_test)
    out2 = pp.transform(df_test)
    pd.testing.assert_frame_equal(out1, out2)