import pandas as pd
import numpy as np
import joblib
import os

# Path
DATA_PATH = "Data/raw/test.csv"
ARTIFACT_DIR = "artifacts"
OUTPUT_PATH = "Data/prediction/submission.csv"

# Load artifacts
preprocessor = joblib.load(os.path.join(ARTIFACT_DIR, "preprocessor.pkl"))
scaler = joblib.load(os.path.join(ARTIFACT_DIR, "scaler.pkl"))
model = joblib.load(os.path.join(ARTIFACT_DIR, "model.pkl"))

# Load Data Test
df_test = pd.read_csv(DATA_PATH)
print(f"Test data loaded: {df_test.shape}")

# Keep Id for submission; drop SalePrice if accidentally present
if "SalePrice" in df_test.columns:
    df_test = df_test.drop(columns=["SalePrice"])

ids = df_test["Id"] if "Id" in df_test.columns else None

# Processor and Scaler
X_test = preprocessor.transform(df_test)
X_test = scaler.transform(X_test)

# Predict
y_log_pred  = model.predict(X_test)
y_pred      = np.expm1(np.clip(y_log_pred, 0, 15))


output = pd.DataFrame({
    "Id": ids,
    "SalePrice": y_pred   
})

# Saving
output.to_csv(OUTPUT_PATH, index=False)

print(" Prediction completed!")
print(f"Predicted SalePrice  — min: {y_pred.min():,.0f}  max: {y_pred.max():,.0f}")
print(f"Saved to: {OUTPUT_PATH}")
