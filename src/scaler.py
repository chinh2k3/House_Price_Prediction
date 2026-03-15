from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd

# BaseEstimator: A base class for all estimators in scikit-learn. It provides basic functionality like parameter management.
# TransformerMixin: A mixin class that provides a default implementation of the fit_transform method.
class SelectiveScaler(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.scaler = StandardScaler()
        self.scale_cols = None

    def fit(self, X: pd.DataFrame, y=None):
        X = X.copy()
        num_cols = X.select_dtypes(include=[np.number]).columns
        flag_cols = [c for c in num_cols if c.endswith("_was_missing") or c.endswith("_nonzero")]
        self.scale_cols = [c for c in num_cols if c not in flag_cols]

        X[self.scale_cols] = X[self.scale_cols].astype(float)

        self.scaler.fit(X[self.scale_cols])
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X[self.scale_cols] = X[self.scale_cols].astype(float)
        X.loc[:, self.scale_cols] = self.scaler.transform(X[self.scale_cols])
        return X
