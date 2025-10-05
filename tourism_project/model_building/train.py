# for data manipulation
import os, json, math
from pathlib import Path
import numpy as np
import pandas as pd

# Scikit-learn
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    average_precision_score,
    roc_auc_score,
    classification_report,
)

# ── Model
import xgboost as xgb
import joblib
import mlflow

#   Hugging Face Hub
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError
from huggingface_hub import login  # if you need to auth via token

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("MLOps_PROD_experiment_1")

api = HfApi()

Xtrain_path = "hf://datasets/cbendale10/MLOps-Tourism-Prediction/Xtrain.csv"
Xtest_path = "hf://datasets/cbendale10/MLOps-Tourism-Prediction/Xtest.csv"
ytrain_path = "hf://datasets/cbendale10/MLOps-Tourism-Prediction/ytrain.csv"
ytest_path = "hf://datasets/cbendale10/MLOps-Tourism-Prediction/ytest.csv"

X_train = pd.read_csv(Xtrain_path)
X_test = pd.read_csv(Xtest_path)
y_train = pd.read_csv(ytrain_path)
y_test = pd.read_csv(ytest_path)


# feature lists from schema
NUMERIC_COLS = [
    "Age","CityTier","NumberOfPersonVisiting","PreferredPropertyStar","NumberOfTrips",
    "Passport","OwnCar","NumberOfChildrenVisiting","MonthlyIncome","PitchSatisfactionScore",
    "NumberOfFollowups","DurationOfPitch"
]
CAT_COLS = [c for c in ["TypeofContact","Occupation","Gender","MaritalStatus","Designation","ProductPitched"] if c in X_train.columns]

preprocessor = make_column_transformer(
    (StandardScaler(), NUMERIC_COLS),
    (OneHotEncoder(handle_unknown="ignore"), CAT_COLS)
)

# class imbalance
pos = int((y_train == 1).sum()); neg = int((y_train == 0).sum())
scale_pos_weight = (neg / max(pos,1)) if pos > 0 else 1.0

xgb_model = xgb.XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1,
    tree_method="hist",
    scale_pos_weight=scale_pos_weight
)

model_pipeline = make_pipeline(preprocessor, xgb_model)

# small, fast grid for dev
param_grid = {
    'xgbclassifier__n_estimators': [60, 80, 100],
    'xgbclassifier__max_depth': [2, 3, 4],
    'xgbclassifier__colsample_bytree': [0.4, 0.5, 0.6],
    'xgbclassifier__colsample_bylevel': [0.4, 0.5, 0.6],
    'xgbclassifier__learning_rate': [0.01, 0.05, 0.1],
    'xgbclassifier__reg_lambda': [0.4, 0.5, 0.6],
}

# -----------------------
# TRAIN + TUNE + LOG
# -----------------------

with mlflow.start_run():
    gs = GridSearchCV(model_pipeline, param_grid=param_grid, cv=5, n_jobs=-1, verbose=0)
    gs.fit(X_train, y_train)

    # log each CV set
    res = gs.cv_results_
    for i in range(len(res["params"])):
        with mlflow.start_run(nested=True):
            mlflow.log_params(res["params"][i])
            mlflow.log_metric("cv_mean", float(res["mean_test_score"][i]))
            mlflow.log_metric("cv_std",  float(res["std_test_score"][i]))

    best_model = gs.best_estimator_
    mlflow.log_params(gs.best_params_)

    # Evaluate at dev threshold
    THRESH = 0.45
    p_tr = best_model.predict_proba(X_train)[:,1]
    p_te = best_model.predict_proba(X_test)[:,1]
    yhat_tr = (p_tr >= THRESH).astype(int)
    yhat_te = (p_te >= THRESH).astype(int)

    def prf(y_true, y_pred):
        p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
        acc = accuracy_score(y_true, y_pred)
        return acc, p, r, f1

    tr_acc, tr_p, tr_r, tr_f1 = prf(y_train, yhat_tr)
    te_acc, te_p, te_r, te_f1 = prf(y_test,  yhat_te)



    metrics = {
    "train_accuracy":  float(tr_acc),
    "train_precision": float(tr_p),
    "train_recall":    float(tr_r),
    "train_f1-score":  float(tr_f1),
    "test_accuracy":   float(te_acc),
    "test_precision":  float(te_p),
    "test_recall":     float(te_r),
    "test_f1-score":   float(te_f1),
    }

    for k, v in metrics.items():
      if v is not None and not (math.isnan(v) or math.isinf(v)):
          mlflow.log_metric(k, v)

    # Save the model locally
    model_path = "best_tourism_prediction_model_v1.joblib"
    joblib.dump(best_model, model_path)

    # Log the model artifact
    mlflow.log_artifact(model_path, artifact_path="model")
    print(f"Model saved as artifact at: {model_path}")

    # Upload to Hugging Face
    repo_id = "cbendale10/MLOps-Tourism-Prediction-model"
    repo_type = "model"

    # Step 1: Check if the space exists
    try:
        api.repo_info(repo_id=repo_id, repo_type=repo_type)
        print(f"Space '{repo_id}' already exists. Using it.")
    except RepositoryNotFoundError:
        print(f"Space '{repo_id}' not found. Creating new space...")
        create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
        print(f"Space '{repo_id}' created.")

    # create_repo("churn-model", repo_type="model", private=False)
    api.upload_file(
        path_or_fileobj="best_machine_failure_model_v1.joblib",
        path_in_repo="best_machine_failure_model_v1.joblib",
        repo_id=repo_id,
        repo_type=repo_type,
    )
