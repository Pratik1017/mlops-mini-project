
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_scorefrom sklearn.metrics import accuracy_score, precision_score, recall_score
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

learning_rate = params["model"]["learning_rate"]
n_estimators = params["model"]["n_estimators"]

models = {
    "LogisticRegression": LogisticRegression(
        max_iter=1000,
        random_state=42
    ),

    "RandomForest": RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=42
    ),

    "GradientBoosting": GradientBoostingClassifier(
        learning_rate=learning_rate,
        n_estimators=n_estimators,
        random_state=42
    )
}

all_metrics = {}

mlflow.set_experiment("Model_Comparison")

with mlflow.start_run(run_name="Multiple_Models_Comparison"):

    for model_name, model in models.items():

        # Nested child run for each model
        with mlflow.start_run(
            run_name=model_name,
            nested=True
        ):

            # Train model
            model.fit(X_train, y_train)

            # Predictions
            y_pred = model.predict(X_test)

            # Calculate metrics
            metrics = {
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(
                    y_test, y_pred, zero_division=0
                ),
                "recall": recall_score(
                    y_test, y_pred, zero_division=0
                ),
            }

            # Log model parameters
            mlflow.log_param("model_name", model_name)

            if model_name == "GradientBoosting":
                mlflow.log_param("learning_rate", learning_rate)

            if model_name in ["RandomForest", "GradientBoosting"]:
                mlflow.log_param("n_estimators", n_estimators)

            # Log metrics
            mlflow.log_metrics(metrics)

            # Log Python script
            mlflow.log_artifact(__file__)

            # Log trained model
            mlflow.sklearn.log_model(
                model,
                name="model",
                skops_trusted_types=[
                    "sklearn.tree._tree.Tree"
                ]
            )

            # Store metrics for DVC
            all_metrics[model_name] = metrics

            print(f"\nModel: {model_name}")
            print(f"Metrics: {metrics}")


# 7. Save all model metrics for DVC
with open("metrics.json", "w") as f:
    json.dump(all_metrics, f, indent=4)

print("\nAll models trained successfully!")
print("Metrics saved to metrics.json")