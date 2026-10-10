
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
import mlflow
import dagshub
import yaml
import json

# 1. Load data
X_train = pd.read_csv("data/external/X_train_processed.csv")
X_test = pd.read_csv("data/external/X_test_processed.csv")
y_train = pd.read_csv("data/processed/y_train.csv").squeeze()
y_test = pd.read_csv("data/processed/y_test.csv").squeeze()

# 2. Initialize MLflow
dagshub.init(
    repo_owner="Pratik1017",
    repo_name="mlops-mini-project",
    mlflow=True
)

mlflow.set_tracking_uri(
    "https://dagshub.com/Pratik1017/mlops-mini-project.mlflow"
)

# 3. Load parameters
with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)

learning_rate = params["models"]["learning_rate"]
n_estimators = params["models"]["n_estimators"]

# 4. Train model and track experiment
with mlflow.start_run():

    clf = GradientBoostingClassifier(
        learning_rate=learning_rate,
        n_estimators=n_estimators
    )

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    # 5. Calculate metrics
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(
            y_test, y_pred, zero_division=0
        ),
        "recall": recall_score(
            y_test, y_pred, zero_division=0
        )
    }

    with open("metric.json","w") as f:
        json.dump(metrics , f)

    # 6. Log parameters and metrics
    mlflow.log_param("learning_rate", learning_rate)
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_metrics(metrics)
    mlflow.log_artifact(__file__)

    # 7. Log model
    mlflow.sklearn.log_model(
        clf,
        artifact_path="model"
    )

    print("Model training completed!")
    print("Metrics:", metrics)