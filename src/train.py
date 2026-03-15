import os
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from processing import Preprocessing
from model import model_train
from scaler import SelectiveScaler


# Path
DATA_PATH = "Data/raw/train.csv"
ARTIFACT_DIR = "artifacts"

# 1. Load Data
df = pd.read_csv('Data/raw/train.csv')
print(f"Loaded data: {df.shape}")

X = df.drop(columns=['SalePrice'])
y = df['SalePrice']

# Log-transform target
y_log = np.log1p(y)


# 2. Train/ validation split
X_raw_train, X_raw_test, y_log_train, y_log_test = train_test_split(X, y_log, test_size = 0.2, random_state = 42)

# Keep raw prices for interpretable metrics
y_raw_train = np.expm1(y_log_train)
y_raw_test = np.expm1(y_log_test)
print(f"Train size: {X_raw_train.shape[0]}  |  Test size: {X_raw_test.shape[0]}")


# 3. Preprocessing
preprocessor = Preprocessing(drop_cols=['Id'])

X_train_pre = preprocessor.fit_transform(X_raw_train)
X_test_pre = preprocessor.transform(X_raw_test)

print(f"After preprocessing — Train: {X_train_pre.shape}  |  Test: {X_test_pre.shape}")

# 4. Scaling
scaler = SelectiveScaler()

X_train = scaler.fit_transform(X_train_pre)
X_test = scaler.transform(X_test_pre)

# 5. Train the model
best_model, best_alpha, best_cv_r2 = model_train(X_train, y_log_train)

print(f"\nBest alpha : {best_alpha:.4f}")
print(f"Best CV R² (log scale): {best_cv_r2:.4f}")

# 6. Evaluate
def evaluate(model, X_tr, X_te, y_tr_raw, y_te_raw):
    y_tr_pred = np.expm1(np.clip(model.predict(X_tr), 0, 15))
    y_te_pred = np.expm1(np.clip(model.predict(X_te), 0, 15))

    def _metrics(y_true, y_pred, label):
        r2 = r2_score(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        rmsle = np.sqrt(mean_squared_error(np.log1p(y_true), np.log1p(np.clip(y_pred, 0, None))))

        print(f"{label}: R2 = {r2:.4f},  MAE = {mae:,.0f},  RMSE = {rmse:,.0f},  RMSLE={rmsle:.4f}")
    
        return r2, mae, rmse, rmsle

    print("Evaluation (raw SalePrice scale)")
    _metrics(y_tr_raw, y_tr_pred, "Train")
    _metrics(y_te_raw, y_te_pred, "Test")

    gap = r2_score(y_tr_raw, y_tr_pred) - r2_score(y_te_raw, y_te_pred)
    if gap > 0.05:
        print(f"Overfit gap: {gap: .4f}")
    
    print(f"Actual  range: [{y_te_raw.min():,.0f} | {y_te_raw.max():,.0f}]")
    print(f"Predicted range: [{y_te_pred.min():,.0f} | {y_te_pred.max():,.0f}]")


evaluate(best_model, X_train, X_test, y_raw_train, y_raw_test)

# 7. Save artifact
os.makedirs(ARTIFACT_DIR, exist_ok = True)

joblib.dump(preprocessor, os.path.join(ARTIFACT_DIR, "preprocessor.pkl"))
joblib.dump(scaler, os.path.join(ARTIFACT_DIR, "scaler.pkl"))
joblib.dump(best_model, os.path.join(ARTIFACT_DIR, "model.pkl"))
print(f"\nArtifacts saved to '{ARTIFACT_DIR}/'")
print("  preprocessor.pkl, scaler.pkl, model.pkl")