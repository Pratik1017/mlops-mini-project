
import pandas as pd
import mlflow
import mlflow.sklearn
import dagshub

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


# 1. Load data
X_train = pd.read_csv("data/external/X_train_processed.csv")
X_test = pd.read_csv("data/external/X_test_processed.csv")

y_train = pd.read_csv("data/processed/y_train.csv").squeeze()
y_test = pd.read_csv("data/processed/y_test.csv").squeeze()


# 2. Initialize DagsHub and MLflow
dagshub.init(
    repo_owner="Pratik1017",
    repo_name="mlops-mini-project",
    mlflow=True
)

mlflow.set_tracking_uri(
    "https://dagshub.com/Pratik1017/mlops-mini-project.mlflow"
)

mlflow.set_experiment("LogisticRegression_Hyperparameter_Tuning")


# 3. Define hyperparameter grid
param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2],
    "max_features": ["sqrt", 0.5],
    "class_weight": [None, "balanced"]
}


# 4. Create model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)


# 5. Cross-validation
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


# 6. GridSearchCV
grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    scoring="f1",
    cv=cv,
    n_jobs=-1,
    refit=True,
    verbose=1
)


# 7. Run experiment
with mlflow.start_run(run_name="RandomForest_GridSearchCV"):

    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_

    # Best parameters
    print("Best Parameters:", grid_search.best_params_)
    print("Best CV F1 Score:", grid_search.best_score_)

    mlflow.log_params(grid_search.best_params_)
    mlflow.log_metric(
        "best_cv_f1",
        grid_search.best_score_
    )

    # 8. Evaluate on test data
    y_pred = best_model.predict(X_test)

    metrics = {
        "test_accuracy": accuracy_score(y_test, y_pred),
        "test_precision": precision_score(
            y_test, y_pred, zero_division=0
        ),
        "test_recall": recall_score(
            y_test, y_pred, zero_division=0
        ),
        "test_f1_score": f1_score(
            y_test, y_pred, zero_division=0
        )
    }

    mlflow.log_metrics(metrics)

    # 9. Log best model

    mlflow.sklearn.log_model(
        best_model,
        name="best_random_forest",
        skops_trusted_types=["sklearn.tree._tree.Tree"]
    )

    print("Test Metrics:", metrics)
    print("Hyperparameter tuning completed!")
