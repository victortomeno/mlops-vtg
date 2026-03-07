import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("diabetes_prueba")

df = pd.read_csv("diabetes.csv")

X = df.drop("Outcome", axis=1)
y = df["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)

with mlflow.start_run(run_name="Regresion_Logistica_Prueba") as run:
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    mlflow.log_metric("accuracy", acc)

    input_example = X_test.iloc[:2]
    
    model_info = mlflow.sklearn.log_model(
        sk_model=model,
        name="modelo_diabetes",
        input_example=input_example
    )

    with open("last_model_uri.txt", "w", encoding="utf-8") as f:
        f.write(model_info.model_uri + "\n")