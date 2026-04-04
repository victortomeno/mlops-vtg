import gradio as gr
import joblib
import numpy as np
from huggingface_hub import hf_hub_download

# 1. Descargar ambos modelos de tu repositorio en Hugging Face
dt_path = hf_hub_download(repo_id="victortomeno/iris26", filename="iris_dt.joblib", repo_type="model")
lr_path = hf_hub_download(repo_id="victortomeno/iris26", filename="iris_logreg.joblib", repo_type="model")

# 2. Cargar los modelos en un diccionario
models = {
    "Decision Tree": joblib.load(dt_path),
    "Logistic Regression": joblib.load(lr_path),
}

LABELS = {0: "Iris-setosa", 1: "Iris-versicolor", 2: "Iris-virginica"}

# 3. Funcion de prediccion
def predict_iris(model_choice, sepal_length, sepal_width, petal_length, petal_width):
    pipeline = models[model_choice]
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = pipeline.predict(input_data)
    return LABELS.get(int(prediction[0]), "Invalid prediction")

# 4. Interfaz de Gradio
interface = gr.Interface(
    fn=predict_iris,
    inputs=[
        gr.Dropdown(choices=["Decision Tree", "Logistic Regression"], label="Model", value="Decision Tree"),
        gr.Number(label="Sepal Length (cm)"),
        gr.Number(label="Sepal Width (cm)"),
        gr.Number(label="Petal Length (cm)"),
        gr.Number(label="Petal Width (cm)"),
    ],
    outputs="text",
    live=True,
    title="Iris Species Identifier",
    description="Elige un modelo e introduce las medidas para predecir la especie de Iris.",
    flagging_mode="manual",
    flagging_dir="flagged"
)

if __name__ == "__main__":
    interface.launch()