"""Data preparation: clean the registered dataset and create stratified train/test splits.

Run from the repository root:  python tourism_project/model_building/prep.py
Outputs Xtrain.csv, Xtest.csv, ytrain.csv, ytest.csv in the current folder; the GitHub
Actions workflow uploads them as the "data-splits" artifact for the training job.
"""
import os

import pandas as pd
from sklearn.model_selection import train_test_split

DATA_PATH = os.path.join("tourism_project", "data", "tourism.csv")
TARGET = "ProdTaken"
DROP_COLUMNS = ["Unnamed: 0", "CustomerID"]   # row index + identifier: no predictive value
TEST_SIZE = 0.20
RANDOM_STATE = 42

# 1. Load the dataset directly from the repository data folder
df = pd.read_csv(DATA_PATH)
print(f"Loaded {DATA_PATH}: {df.shape[0]:,} rows x {df.shape[1]} columns")

# 2. Remove unnecessary columns
df = df.drop(columns=[c for c in DROP_COLUMNS if c in df.columns])
print(f"Dropped columns: {DROP_COLUMNS}")

# 3. Fix inconsistent category labels ("Fe Male" is a typo of "Female")
n_fixed = int((df["Gender"] == "Fe Male").sum())
df["Gender"] = df["Gender"].replace("Fe Male", "Female")
print(f"Fixed 'Fe Male' -> 'Female' in {n_fixed} rows")

# 4. Remove exact duplicate records so the same profile cannot land in both train and test
n_before = len(df)
df = df.drop_duplicates().reset_index(drop=True)
print(f"Removed {n_before - len(df)} duplicate rows")

# 5. Impute missing values (none today, but future data may have gaps)
df = df.dropna(subset=[TARGET])
num_cols = df.select_dtypes(include="number").columns.drop(TARGET)
cat_cols = df.select_dtypes(include="object").columns
n_missing = int(df.isnull().sum().sum())
df[num_cols] = df[num_cols].fillna(df[num_cols].median())
for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])
print(f"Imputed {n_missing} missing values")

# 6. Stratified train/test split (keeps the ~19% purchase rate in both sets)
X = df.drop(columns=[TARGET])
y = df[TARGET]
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
)

# 7. Save the splits locally - the workflow passes them to the next job as an artifact
Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)

print(f"\nClean dataset : {df.shape[0]:,} rows x {df.shape[1]} columns")
print(f"Train split   : {Xtrain.shape[0]:,} rows (purchase rate {ytrain.mean():.1%})")
print(f"Test split    : {Xtest.shape[0]:,} rows (purchase rate {ytest.mean():.1%})")
print("Saved Xtrain.csv, Xtest.csv, ytrain.csv, ytest.csv")
