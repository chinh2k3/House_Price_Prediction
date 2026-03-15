import pandas as pd
import sys
from pathlib import Path 

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from src.processing import Preprocessing
def run_test():
    train_df = pd.read_csv("Data/raw/train.csv")
    test_df  = pd.read_csv("Data/raw/test.csv")

    pre = Preprocessing(drop_cols=["Id", "SalePrice"])

    X_train = pre.fit_transform(train_df)
    X_test  = pre.transform(test_df)

    print("Train shape:", X_train.shape)
    print("Test shape :", X_test.shape)

    assert X_train.shape[1] == X_test.shape[1], "Shape mismatch"

    assert X_train.isnull().sum().sum() == 0, "NaN in train"
    assert X_test.isnull().sum().sum() == 0, "NaN in test"

    for col in ["TotalArea","HouseAge","YearsSinceRemod","Remodeled","HasLotFrontage"]:
        assert col in X_train.columns, f"Missing feature: {col}"

    print("✔ Preprocessing test PASSED")

if __name__ == "__main__":
    run_test()
