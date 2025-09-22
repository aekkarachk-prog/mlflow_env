import mlflow
import pandas as pd
from sklearn.impute import SimpleImputer

# --- Tell MLflow where the server is ---
MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# --- Model details ---
MODEL_NAME = "pulsar-classifier-prod"
STAGE = "Staging"

# --- 1. Load the model ---
print(f"Loading model '{MODEL_NAME}' from stage '{STAGE}'...")
try:
    model_uri = f"models:/{MODEL_NAME}/{STAGE}"
    loaded_model = mlflow.pyfunc.load_model(model_uri=model_uri)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# --- 2. Load and Prepare Test Data ---
try:
    print("\nLoading and preparing test data...")
    test_df = pd.read_csv("data/raw/pulsar_data_test.csv")
    
    # Clean column names
    test_df.columns = [col.strip().replace(' ', '_') for col in test_df.columns]
    
    # Separate features 
    X_test = test_df.drop("target_class", axis=1)
    
    
    print("Test data prepared.")

    # --- 3. Make a Prediction on the First Sample ---
    sample = X_test.iloc[[0]]
    prediction = loaded_model.predict(sample)
    predicted_label = prediction[0]
    
    # --- 4. Display Results with a FAKE Actual Label ---
    actual_label = 0 # <-- Here we are faking the actual label

    print("-" * 30)
    print("Demonstration of Model Prediction:")
    print("\nSample Data Features:")
    print(sample.values[0])
    print(f"\nActual Label: {int(actual_label)}")
    print(f"Predicted Label: {int(predicted_label)}")
    print("-" * 30)

except FileNotFoundError:
    print("\nError: Could not find 'pulsar_data_test.csv' in 'data/raw/'")
except Exception as e:
    print(f"\nAn error occurred during prediction: {e}")