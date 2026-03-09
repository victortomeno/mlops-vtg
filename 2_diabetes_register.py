import mlflow
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri("sqlite:///mlflow.db")

def read_last_model_uri(path="last_model_uri.txt") -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.readline().strip()

if __name__ == "__main__":
    model_uri = read_last_model_uri()
    model_name = "Diabetes_LR_Model"
    
    result = mlflow.register_model(model_uri=model_uri, name=model_name)
    
    print(f"Modelo registrado: {result.name}, version: {result.version}")
    
    client = MlflowClient()
    model_version_info = client.get_model_version(name=result.name, version=result.version)
    print("Estado del modelo registrado:", model_version_info.status)