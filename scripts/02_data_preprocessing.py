import os
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.impute import SimpleImputer
import mlflow

def preprocess_pulsar_data():
    """
    Loads raw pulsar data, cleans it, handles missing values, 
    balances the classes using SMOTE, and logs the processed 
    training data as an artifact in MLflow.
    """
    # --- 1. Set up MLflow Tracking ---
    # This logic uses the environment variable in GitHub Actions, 
    # but falls back to a local server for development.
    MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("Pulsar Star - Data Pipeline")

    with mlflow.start_run() as run:
        run_id = run.info.run_id
        print(f"Starting data preprocessing run with Run ID: {run_id}")
        mlflow.set_tag("ml.step", "data_preprocessing")

        # --- 2. Load Raw Data ---
        RAW_DATA_PATH = "data/raw/pulsar_data_train.csv"
        print(f"Loading data from {RAW_DATA_PATH}...")
        df = pd.read_csv(RAW_DATA_PATH)
        
        # --- 3. Data Cleaning and Preparation ---
        print("Cleaning and preparing data...")
        # Clean column names
        df.columns = [col.strip().replace(' ', '_') for col in df.columns]
        
        # Drop rows where the target is missing
        initial_rows = len(df)
        df.dropna(subset=['target_class'], inplace=True)
        if initial_rows > len(df):
            print(f"Dropped {initial_rows - len(df)} rows with missing target.")

        X = df.drop("target_class", axis=1)
        y = df["target_class"]

        # Handle missing values in features (X)
        if X.isnull().sum().sum() > 0:
            imputer = SimpleImputer(strategy='mean')
            X_imputed = imputer.fit_transform(X)
            X = pd.DataFrame(X_imputed, columns=X.columns)
        
        # --- 4. Balance Data with SMOTE ---
        print("Balancing data with SMOTE...")
        rows_before_smote = len(X)
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)
        rows_after_smote = len(X_resampled)
        
        # --- 5. Save Processed Data Locally ---
        processed_data_dir = "processed_data"
        os.makedirs(processed_data_dir, exist_ok=True)
        
        balanced_df = pd.concat([X_resampled, y_resampled], axis=1)
        balanced_df.to_csv(os.path.join(processed_data_dir, "train_balanced.csv"), index=False)
        print(f"Saved balanced data to '{processed_data_dir}' directory.")

        # --- 6. Log Parameters, Metrics, and Artifacts to MLflow ---
        mlflow.log_param("smote_random_state", 42)
        mlflow.log_metric("rows_before_smote", rows_before_smote)
        mlflow.log_metric("rows_after_smote", rows_after_smote)
        
        mlflow.log_artifacts(processed_data_dir, artifact_path="processed_data")
        print("Logged processed data directory as an artifact in MLflow.")

        # --- 7. Output Run ID for CI/CD ---
        print("-" * 50)
        print(f"Data preprocessing complete. Use this Run ID for the training step:")
        print(f"Preprocessing Run ID: {run_id}")
        print("-" * 50)

        if "GITHUB_OUTPUT" in os.environ:
            with open(os.environ["GITHUB_OUTPUT"], "a") as f:
                print(f"run_id={run_id}", file=f)

if __name__ == "__main__":
    preprocess_pulsar_data()
