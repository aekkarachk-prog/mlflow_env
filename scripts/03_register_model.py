# scripts/03_register_model.py
import mlflow
from mlflow.tracking import MlflowClient

MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# ชื่อ Experiment และ Model ที่จะลงทะเบียน
EXPERIMENT_NAME = "pulsar-star-classifier"
REGISTERED_MODEL_NAME = "pulsar-classifier-prod"

# ดึงข้อมูล Experiment
experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
if not experiment:
    raise RuntimeError(f"Experiment '{EXPERIMENT_NAME}' not found.")

# ค้นหา Run ที่ดีที่สุด (เรียงจาก f1_score สูงสุด)
best_run = client.search_runs(
    experiment_ids=experiment.experiment_id,
    order_by=["metrics.f1_score DESC"],
    max_results=1
)[0]

run_id = best_run.info.run_id
model_uri = f"runs:/{run_id}/pulsar-classifier-model"

print(f"Found best run with F1 Score: {best_run.data.metrics['f1_score']:.4f}")
print(f"Run ID: {run_id}")
print(f"Model URI: {model_uri}")

# ลงทะเบียนโมเดล
print(f"Registering model '{REGISTERED_MODEL_NAME}'...")
model_version = mlflow.register_model(
    model_uri=model_uri,
    name=REGISTERED_MODEL_NAME
)

print(f"Model registered successfully as Version: {model_version.version}")

# ย้ายโมเดลเวอร์ชันล่าสุดไปที่ Stage 'Staging'
print("Transitioning model to 'Staging'...")
client.transition_model_version_stage(
    name=REGISTERED_MODEL_NAME,
    version=model_version.version,
    stage="Staging"
)
print("Transition complete.")