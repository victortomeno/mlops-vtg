import pandas as pd
import requests
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

df = pd.read_csv("diabetes.csv")
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# Separacion del 10% del dataset original para la fase de test
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.10, random_state=42)

data_split = X_test.to_dict(orient="split")
del data_split["index"]

payload = {
    "dataframe_split": data_split
}

url = "http://localhost:5000/invocations"
headers = {"Content-Type": "application/json"}

response = requests.post(url, headers=headers, data=json.dumps(payload))

if response.status_code == 200:
    preds = response.json()
    predictions_list = preds.get("predictions", preds) if isinstance(preds, dict) else preds
    
    # Calculo de las metricas finales
    acc = accuracy_score(y_test, predictions_list)
    prec = precision_score(y_test, predictions_list, zero_division=0)
    rec = recall_score(y_test, predictions_list, zero_division=0)
    f1 = f1_score(y_test, predictions_list, zero_division=0)
    
    print("=== METRICAS FINALES SOBRE DATASET DE TEST (10%) ===")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 Score:  {f1:.4f}")
else:
    print(f"Error HTTP: {response.status_code}")
    print(response.text)