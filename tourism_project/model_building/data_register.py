"""Data registration: validate the tourism dataset stored in the repository and print a summary.

Run from the repository root:  python tourism_project/model_building/data_register.py
Exits with a non-zero status if validation fails, which stops the GitHub Actions pipeline.
"""
import hashlib
import os
import sys

import pandas as pd

DATA_PATH = os.path.join("tourism_project", "data", "tourism.csv")
TARGET = "ProdTaken"
MIN_ROWS = 1000

# Every column the downstream scripts rely on
EXPECTED_COLUMNS = [
    "CustomerID", "ProdTaken", "Age", "TypeofContact", "CityTier", "DurationOfPitch",
    "Occupation", "Gender", "NumberOfPersonVisiting", "NumberOfFollowups", "ProductPitched",
    "PreferredPropertyStar", "MaritalStatus", "NumberOfTrips", "Passport",
    "PitchSatisfactionScore", "OwnCar", "NumberOfChildrenVisiting", "Designation",
    "MonthlyIncome",
]


def file_md5(path):
    """MD5 fingerprint of the file - used as the dataset version."""
    md5 = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            md5.update(chunk)
    return md5.hexdigest()


def main():
    print("=" * 60)
    print("DATA REGISTRATION")
    print("=" * 60)

    # 1. The dataset must exist in the repo data folder
    if not os.path.exists(DATA_PATH):
        sys.exit(f"[FAIL] Dataset not found at {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    print(f"[OK]   Loaded {DATA_PATH}")

    # 2. All expected columns must be present
    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing:
        sys.exit(f"[FAIL] Missing expected columns: {missing}")
    print(f"[OK]   All {len(EXPECTED_COLUMNS)} expected columns present")

    extra = [c for c in df.columns if c not in EXPECTED_COLUMNS]
    if extra:
        print(f"[INFO] Extra columns (dropped later in prep.py): {extra}")

    # 3. Target must be binary 0/1 and the dataset must be large enough to train on
    if not set(df[TARGET].dropna().unique()) <= {0, 1}:
        sys.exit(f"[FAIL] {TARGET} must only contain 0/1")
    print(f"[OK]   Target '{TARGET}' is binary")
    if len(df) < MIN_ROWS:
        sys.exit(f"[FAIL] Only {len(df)} rows - need at least {MIN_ROWS}")
    print(f"[OK]   Row count {len(df):,} >= {MIN_ROWS:,}")

    # 4. Summary
    print("\n--- Dataset summary ---")
    print(f"Version (MD5)     : {file_md5(DATA_PATH)}")
    print(f"Rows x columns    : {df.shape[0]:,} x {df.shape[1]}")
    print(f"Numerical columns : {df.select_dtypes('number').shape[1]}")
    print(f"Text columns      : {df.select_dtypes('object').shape[1]}")
    print(f"Missing values    : {int(df.isnull().sum().sum())}")
    print(f"Duplicate IDs     : {int(df['CustomerID'].duplicated().sum())}")
    rate = df[TARGET].value_counts(normalize=True).sort_index()
    print(f"Target balance    : 0 = {rate.get(0, 0):.1%} | 1 = {rate.get(1, 0):.1%}")
    print("\nDataset registered successfully.")


if __name__ == "__main__":
    main()
