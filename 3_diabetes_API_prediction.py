import pandas as pd
import requests
import json
from sklearn.model_selection import train_test_split

df = pd.read_csv("diabetes.csv")
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

_, X_test, _, y_test = train_test_split(X, y, test_size=0.10, random_state=42)

data_split = X_test.iloc[:5].to_dict(orient="split")
del data_split["index"]

payload = {
    "dataframe_split": data_split
}

url = "http://localhost:5000/invocations"
headers = {"Content-Type": "application/json"}

response = requests.post(url, headers=headers, data=json.dumps(payload))

if response.status_code == 200:
    preds = response.json()
    print("=== PREDICCIONES VS. VALORES REALES ===")
    
    predictions_list = preds.get("predictions", preds) if isinstance(preds, dict) else preds

    for idx, (pred, real) in enumerate(zip(predictions_list, y_test.iloc[:5])):
        print(f"Fila {idx + 1} - Prediccion: {pred} | Real: {real}")
else:
    print(f"Error HTTP: {response.status_code}")
    print(response.text)