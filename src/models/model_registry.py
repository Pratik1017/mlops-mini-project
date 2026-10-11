
import dagshub
import mlflow
from mlflow.tracking import MlflowClient

# DagsHub + MLflow setup
dagshub.init(
    repo_owner="Pratik1017",
    repo_name="mlops-mini-project",
    mlflow=True
)

client = MlflowClient()

# Your best Random Forest tuning run
run_id = "0aed95949bf741e1afbf522edfa6d579"

# Artifact name used when logging the model
model_uri = f"runs:/{run_id}/best_random_forest"

# Register model
model_name = "Best_RandomForest_Model"

registered_model = mlflow.register_model(
    model_uri=model_uri,
    name=model_name
)

version = registered_model.version

# Assign alias to this model version
client.set_registered_model_alias(
    name=model_name,
    alias="champion",
    version=version
)

print(f"Model registered: {model_name}")
print(f"Model version: {version}")
print("Alias assigned: champion")
