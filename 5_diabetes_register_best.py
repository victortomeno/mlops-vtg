import mlflow

mlflow.set_tracking_uri("sqlite:///mlflow.db")

run_id = "58bab58714144994b608d3d93939deb0"
model_uri = f"runs:/{run_id}/modelo"
model_name = "Diabetes_Best_Model"

result = mlflow.register_model(model_uri=model_uri, name=model_name)

print(f"Modelo registrado: {result.name}, version: {result.version}")