# Visit with Us - Wellness Tourism Package Prediction (MLOps)

End-to-end MLOps pipeline that predicts whether a customer will purchase the **Wellness Tourism Package**
before the sales team contacts them. Every push to `main` retrains the model with GitHub Actions and
commits the best model back to this repository, where the Streamlit app picks it up.

## Repository structure

```
tourism_project/
├── data/tourism.csv                 # registered dataset
├── model_building/
│   ├── data_register.py             # validates the dataset and prints a summary
│   ├── prep.py                      # cleaning + stratified train/test split
│   └── train.py                     # XGBoost + GridSearchCV + MLflow tracking
├── deployment/
│   ├── app.py                       # Streamlit app
│   ├── requirements.txt             # app dependencies
│   ├── best_model.joblib            # committed by the pipeline
│   └── model_metadata.json          # metrics of the committed model
└── requirements.txt                 # pipeline dependencies
.github/workflows/pipeline.yml       # CI/CD pipeline
```

## Pipeline (GitHub Actions)

`register-dataset` → `data-prep` → `model-traning`

1. **register-dataset** - validates the CSV (expected columns, binary target) and uploads it as an artifact.
2. **data-prep** - drops ID columns, fixes label typos, removes duplicates, imputes, and splits 80/20 (stratified); uploads the splits as an artifact.
3. **model-traning** - downloads the splits, starts an MLflow server, tunes XGBoost with GridSearchCV (every combination logged as a nested MLflow run), evaluates on the test set, and commits `best_model.joblib` to `main`.

## Run locally

```bash
pip install -r tourism_project/requirements.txt
python tourism_project/model_building/data_register.py
python tourism_project/model_building/prep.py
python tourism_project/model_building/train.py
pip install -r tourism_project/deployment/requirements.txt
streamlit run tourism_project/deployment/app.py
```
