# scripts/02_train_model.py
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import mlflow
import os

# --- Set up MLflow Tracking ---
# This logic checks for a tracking URI from the GitHub Actions environment variable first.
# If it's not found, it falls back to your local server for development.
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# --- The rest of your script remains exactly the same ---
mlflow.set_experiment("pulsar-star-classifier")

# อ่านข้อมูล Train
train_df = pd.read_csv("data/processed/train_balanced.csv")
X_train = train_df.drop("target_class", axis=1)
y_train = train_df["target_class"]

# เริ่มต้น MLflow run
with mlflow.start_run():
    # --- Parameters ---
    n_estimators = 150
    max_depth = 10
    random_state = 42
    
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_param("random_state", random_state)

    # สร้าง Pipeline: StandardScaler -> RandomForestClassifier
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state
        ))
    ])

    # Train โมเดล
    print("Training model...")
    pipeline.fit(X_train, y_train)

    # ประเมินผลโมเดล (ใช้ข้อมูล Train เพื่อความง่าย)
    y_pred = pipeline.predict(X_train)
    
    # --- Metrics ---
    accuracy = accuracy_score(y_train, y_pred)
    precision = precision_score(y_train, y_pred)
    recall = recall_score(y_train, y_pred)
    f1 = f1_score(y_train, y_pred)
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)
    
    # --- Model ---
    # Log the model (pipeline)
    mlflow.sklearn.log_model(pipeline, "pulsar-classifier-model")

    # Get the run_id of the current run
    run_id = mlflow.active_run().info.run_id
    print("\nModel training and logging complete.")
    print(f"Run ID: {run_id}")

    # --- GITHUB ACTIONS INTEGRATION --- #
    # This block writes the run_id to a special file that GitHub Actions can read.
    # It only runs when the script is executed within a GitHub Actions environment.
    print("Checking for GITHUB_OUTPUT environment variable...")
    if "GITHUB_OUTPUT" in os.environ:
        print(f"Writing run_id {run_id} to GITHUB_OUTPUT...")
        with open(os.environ["GITHUB_OUTPUT"], "a") as f:
            print(f"run_id={run_id}", file=f)
        print("Successfully wrote run_id to GITHUB_OUTPUT.")
