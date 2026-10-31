"""Model training with hyperparameter tuning and MLflow experiment tracking.

Run from the repository root (after prep.py):  python tourism_project/model_building/train.py
Saves the best model to tourism_project/deployment/best_model.joblib for the workflow to commit.
"""
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import joblib
import mlflow
import pandas as pd
import xgboost as xgb
from sklearn.compose import make_column_transformer
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_STATE = 42
DEPLOYMENT_DIR = Path("tourism_project") / "deployment"
MODEL_PATH = DEPLOYMENT_DIR / "best_model.joblib"
METADATA_PATH = DEPLOYMENT_DIR / "model_metadata.json"

# ---------------------------------------------------------------- MLflow setup
# GitHub Actions sets MLFLOW_TRACKING_URI to the MLflow server started in the workflow;
# locally we fall back to a file store in ./mlruns.
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", Path("mlruns").resolve().as_uri()))
mlflow.set_experiment("tourism-package-prediction")
print(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")

# ---------------------------------------------------------------- Load the splits
Xtrain = pd.read_csv("Xtrain.csv")
Xtest = pd.read_csv("Xtest.csv")
ytrain = pd.read_csv("ytrain.csv").squeeze("columns")
ytest = pd.read_csv("ytest.csv").squeeze("columns")
print(f"Train: {Xtrain.shape} | Test: {Xtest.shape}")

numeric_features = [
    "Age", "CityTier", "DurationOfPitch", "NumberOfPersonVisiting", "NumberOfFollowups",
    "PreferredPropertyStar", "NumberOfTrips", "Passport", "PitchSatisfactionScore",
    "OwnCar", "NumberOfChildrenVisiting", "MonthlyIncome",
]
categorical_features = [
    "TypeofContact", "Occupation", "Gender", "ProductPitched", "MaritalStatus", "Designation",
]

# ---------------------------------------------------------------- Model definition
# Scale numbers and one-hot encode categories; unseen categories at prediction time are ignored.
preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (OneHotEncoder(handle_unknown="ignore"), categorical_features),
)

# Weight the minority (purchase) class by the negative/positive ratio (~4:1)
scale_pos_weight = round((ytrain == 0).sum() / (ytrain == 1).sum(), 2)
xgb_model = xgb.XGBClassifier(
    objective="binary:logistic", eval_metric="logloss",
    scale_pos_weight=scale_pos_weight, random_state=RANDOM_STATE, n_jobs=-1,
)
model_pipeline = make_pipeline(preprocessor, xgb_model)

# Hyperparameter grid (keys use the make_pipeline step name "xgbclassifier")
param_grid = {
    "xgbclassifier__n_estimators": [100, 200, 300],
    "xgbclassifier__max_depth": [3, 5, 7],
    "xgbclassifier__learning_rate": [0.05, 0.1],
    "xgbclassifier__subsample": [0.8, 1.0],
    "xgbclassifier__colsample_bytree": [0.7, 1.0],
}
n_combinations = 1
for values in param_grid.values():
    n_combinations *= len(values)
print(f"Tuning {n_combinations} parameter combinations with 5-fold CV (scale_pos_weight={scale_pos_weight})")

grid_search = GridSearchCV(
    model_pipeline, param_grid,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
    scoring="f1", n_jobs=-1, return_train_score=True,
)


def evaluate(model, X, y):
    """Classification metrics for the purchase class (ProdTaken = 1)."""
    pred = model.predict(X)
    proba = model.predict_proba(X)[:, 1]
    return {
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred),
        "recall": recall_score(y, pred),
        "f1": f1_score(y, pred),
        "roc_auc": roc_auc_score(y, proba),
    }


with mlflow.start_run(run_name=f"xgb-gridsearch-{datetime.now(timezone.utc):%Y%m%d-%H%M%S}") as run:
    mlflow.set_tags({"model_type": "XGBClassifier", "tuning": "GridSearchCV",
                     "trigger": os.getenv("GITHUB_EVENT_NAME", "local")})
    mlflow.log_params({"scale_pos_weight": scale_pos_weight, "cv_folds": 5,
                       "scoring": "f1", "n_combinations": n_combinations,
                       "train_rows": len(Xtrain), "test_rows": len(Xtest)})

    # ------------------------------------------------------------ Tune the model
    grid_search.fit(Xtrain, ytrain)

    # Log every tuned parameter combination as a nested run
    results = grid_search.cv_results_
    for i in range(len(results["params"])):
        with mlflow.start_run(run_name=f"combo-{i:02d}", nested=True):
            mlflow.log_params({k.replace("xgbclassifier__", ""): v
                               for k, v in results["params"][i].items()})
            mlflow.log_metrics({
                "mean_cv_f1": results["mean_test_score"][i],
                "std_cv_f1": results["std_test_score"][i],
                "mean_train_f1": results["mean_train_score"][i],
            })

    # ------------------------------------------------------------ Evaluate the best model
    best_model = grid_search.best_estimator_
    best_params = {k.replace("xgbclassifier__", ""): v for k, v in grid_search.best_params_.items()}
    mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})
    mlflow.log_metric("best_cv_f1", grid_search.best_score_)

    train_metrics = evaluate(best_model, Xtrain, ytrain)
    test_metrics = evaluate(best_model, Xtest, ytest)
    mlflow.log_metrics({f"train_{k}": v for k, v in train_metrics.items()})
    mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})

    report = classification_report(ytest, best_model.predict(Xtest),
                                   target_names=["Not purchased", "Purchased"], digits=3)
    cm = confusion_matrix(ytest, best_model.predict(Xtest))
    mlflow.log_text(report, "classification_report.txt")

    print(f"\nBest parameters : {best_params}")
    print(f"Best CV F1      : {grid_search.best_score_:.4f}\n")
    print(pd.DataFrame({"train": train_metrics, "test": test_metrics}).round(4).to_string())
    print(f"\nClassification report (test set):\n{report}")
    print(f"Confusion matrix (test set) [rows = actual, cols = predicted]:\n{cm}")

    # ------------------------------------------------------------ Save the best model
    DEPLOYMENT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, MODEL_PATH)
    metadata = {
        "model": "XGBClassifier (scikit-learn pipeline)",
        "trained_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "mlflow_run_id": run.info.run_id,
        "best_params": best_params,
        "best_cv_f1": round(grid_search.best_score_, 4),
        "test_metrics": {k: round(v, 4) for k, v in test_metrics.items()},
        "features": list(Xtrain.columns),
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2, default=str))
    mlflow.log_artifact(str(MODEL_PATH), artifact_path="model")
    mlflow.log_artifact(str(METADATA_PATH), artifact_path="model")

    print(f"\nSaved best model to {MODEL_PATH} ({MODEL_PATH.stat().st_size / 1024:.0f} KB)")
    print(f"MLflow run id: {run.info.run_id}")
