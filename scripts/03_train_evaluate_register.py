

import mlflow
import os
import sys

def register_model():
    """
    Registers the model from a specific run_id and promotes it
    by giving it the 'champion' alias.
    """
    # --- 1. Set up MLflow Tracking ---
    # Reads the tracking URI from the environment variable in GitHub Actions
    MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
    if MLFLOW_TRACKING_URI is None:
        print("Error: MLFLOW_TRACKING_URI environment variable not set.")
        sys.exit(1)
    
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = mlflow.tracking.MlflowClient()

    # --- 2. Get Run ID from command-line arguments ---
    if len(sys.argv) < 2:
        print("Error: Missing required argument: <run_id>")
        print("Usage: python scripts/03_train_evaluate_register.py <run_id>")
        sys.exit(1)
        
    run_id = sys.argv[1]
    print(f"Received Run ID: {run_id}")

    # --- 3. Register the Model ---
    model_name = "pulsar-classifier-prod"
    model_uri = f"runs:/{run_id}/pulsar-classifier-model"
    
    try:
        print(f"Registering model '{model_name}' from URI: {model_uri}")
        model_version = mlflow.register_model(
            model_uri=model_uri,
            name=model_name
        )
        print(f"Model registered successfully as Version: {model_version.version}")

        # --- 4. Promote Model using Alias ---
        # Set the 'champion' alias for the newly registered version
        print(f"Setting 'champion' alias for model version {model_version.version}...")
        client.set_registered_model_alias(
            name=model_name,
            alias="champion",
            version=model_version.version
        )
        print("Successfully set 'champion' alias.")

    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    register_model()