import os
import sys
import pandas as pd
import mlflow
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score

def train_and_register():
    """
    Loads processed data from a previous MLflow run, trains a model
    in a new run, logs it, and then registers the new model.
    """
    # --- 1. Set up MLflow Tracking ---



    # --- 2. Get Preprocessing Run ID from Arguments ---
    if len(sys.argv) < 2:
        print("Error: Missing required argument: <preprocessing_run_id>")
        sys.exit(1)
    preprocessing_run_id = sys.argv[1]
    print(f"Using preprocessing data from Run ID: {preprocessing_run_id}")


    # --- 3. Load Data Artifact from the Preprocessing Run ---
    try:
        data_path = f"runs:/{preprocessing_run_id}/data/processed_data/train_balanced.csv"
        print(f"Loading data from artifact path: {data_path}")
        train_df = pd.read_csv(data_path)
        X_train = train_df.drop("target_class", axis=1)
        y_train = train_df["target_class"]
    except Exception as e:
        print(f"Error loading data artifact: {e}")
        sys.exit(1)


    # --- 4. Start a NEW MLflow Run for Training ---
    mlflow.set_experiment("Pulsar Star - Model Training")
    with mlflow.start_run() as run:
        training_run_id = run.info.run_id
        print(f"Started new training run with Run ID: {training_run_id}")
        mlflow.set_tag("ml.step", "model_training")
        mlflow.set_tag("parent.run_id", preprocessing_run_id) # Link to parent run

        # Log parameters
        n_estimators = 150
        max_depth = 10
        mlflow.log_params({"n_estimators": n_estimators, "max_depth": max_depth})

        # Create and train the model pipeline
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42))
        ])
        pipeline.fit(X_train, y_train)

        # Evaluate and log metrics
        y_pred = pipeline.predict(X_train)
        f1 = f1_score(y_train, y_pred)
        mlflow.log_metric("f1_score", f1)
        print(f"Model F1 Score: {f1:.4f}")

        # Log the model artifact
        model_artifact_path = "pulsar-classifier-model"
        mlflow.sklearn.log_model(pipeline, model_artifact_path)
        print(f"Logged model artifact as '{model_artifact_path}'")


        # --- 5. Register the Model from THIS Run ---
        model_name = "pulsar-classifier-prod"
        model_uri = f"runs:/{training_run_id}/{model_artifact_path}"
        print(f"Registering model '{model_name}' from URI: {model_uri}")
        
        model_version = mlflow.register_model(model_uri=model_uri, name=model_name)
        print(f"Model registered successfully as Version: {model_version.version}")


        # --- 6. Output Training Run ID for CI/CD ---
        if "GITHUB_OUTPUT" in os.environ:
            with open(os.environ["GITHUB_OUTPUT"], "a") as f:
                print(f"run_id={training_run_id}", file=f)
                print(f"model_version={model_version.version}", file=f)

if __name__ == "__main__":
    train_and_register()