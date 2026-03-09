import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, ParameterGrid
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("diabetes_experimentos_completos")

df = pd.read_csv("diabetes.csv")
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.10, random_state=42)

X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=(0.20/0.90), random_state=42)

param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],
    'solver': ['liblinear', 'lbfgs'],
    'max_iter': [1000]
}

grid = ParameterGrid(param_grid)

for params in grid:
    with mlflow.start_run():
        model = LogisticRegression(**params)
        model.fit(X_train, y_train)
        
        y_pred_val = model.predict(X_val)
        
        acc = accuracy_score(y_val, y_pred_val)
        prec = precision_score(y_val, y_pred_val, zero_division=0)
        rec = recall_score(y_val, y_pred_val, zero_division=0)
        f1 = f1_score(y_val, y_pred_val, zero_division=0)
        
        mlflow.log_params(params)
        mlflow.log_metrics({
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1
        })
        
        input_example = X_val.iloc[:2]
        mlflow.sklearn.log_model(sk_model=model, name="modelo", input_example=input_example)

print("Experimentos completados y registrados en MLflow.")