"""
app-iris-ct: Continuous Training extension for ML-FastAPI-Docker
================================================================
Extends app-iris with:
  - POST /train      → reentrenamiento incremental con nuevas muestras
  - GET  /model/info → versión activa, métricas, historial
  - POST /predict    → inferencia (igual que app-iris, con versión activa)
  - GET  /health     → estado del servicio
"""

import json
import os
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Configuración de rutas
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

MODEL_PATH = MODELS_DIR / "model_active.joblib"
HISTORY_PATH = MODELS_DIR / "training_history.json"

# ---------------------------------------------------------------------------
# Esquemas Pydantic
# ---------------------------------------------------------------------------

class IrisSample(BaseModel):
    sepal_length: float = Field(..., example=5.1, description="Longitud del sépalo (cm)")
    sepal_width: float  = Field(..., example=3.5, description="Anchura del sépalo (cm)")
    petal_length: float = Field(..., example=1.4, description="Longitud del pétalo (cm)")
    petal_width: float  = Field(..., example=0.2, description="Anchura del pétalo (cm)")


class LabeledSample(BaseModel):
    sepal_length: float = Field(..., example=5.1)
    sepal_width: float  = Field(..., example=3.5)
    petal_length: float = Field(..., example=1.4)
    petal_width: float  = Field(..., example=0.2)
    label: int = Field(..., ge=0, le=2, example=0,
                       description="0=setosa, 1=versicolor, 2=virginica")


class PolicyConfig(BaseModel):
    method: str = Field("any_improvement", description="any_improvement, min_delta, per_class_f1")
    threshold: float = Field(0.02, description="Umbral para min_delta")
    target_class: int = Field(0, description="Clase objetivo para per_class_f1")


class TrainRequest(BaseModel):
    samples: List[LabeledSample] = Field(
        ..., min_items=5,
        description="Nuevas muestras etiquetadas para reentrenamiento (mínimo 5)"
    )
    retrain_from_scratch: bool = Field(
        False,
        description="Si True, ignora datos anteriores y entrena solo con las muestras enviadas"
    )
    policy: PolicyConfig = Field(default_factory=PolicyConfig)


class PredictResponse(BaseModel):
    prediction: int
    class_name: str
    model_version: str


class TrainResponse(BaseModel):
    status: str
    model_version: str
    accuracy_new: float
    accuracy_previous: Optional[float]
    model_updated: bool
    message: str


class ModelInfo(BaseModel):
    active_version: str
    trained_at: str
    accuracy: float
    n_training_samples: int
    algorithm: str
    history: List[dict]


# ---------------------------------------------------------------------------
# Utilidades de persistencia
# ---------------------------------------------------------------------------

CLASS_NAMES = {0: "setosa", 1: "versicolor", 2: "virginica"}


def load_history() -> List[dict]:
    if HISTORY_PATH.exists():
        with open(HISTORY_PATH) as f:
            return json.load(f)
    return []


def save_history(history: List[dict]):
    with open(HISTORY_PATH, "w") as f:
        json.dump(history, f, indent=2)


def get_active_model_meta() -> Optional[dict]:
    history = load_history()
    active_entries = [h for h in history if h.get("activated", True)]
    return active_entries[-1] if active_entries else (history[-1] if history else None)


# ---------------------------------------------------------------------------
# Bootstrap: si no existe modelo, lo entrenamos con el dataset original
# ---------------------------------------------------------------------------

def bootstrap_model():
    """Entrena un modelo base con el dataset Iris completo al arrancar."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    iris = load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    clf = LogisticRegression(max_iter=200, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = float(accuracy_score(y_test, y_pred))
    f1_scores = [round(float(f), 4) for f in f1_score(y_test, y_pred, average=None, labels=[0, 1, 2])]

    version = "v1.0-base"
    joblib.dump(clf, MODEL_PATH)

    history = [{
        "version": version,
        "trained_at": datetime.utcnow().isoformat() + "Z",
        "accuracy": round(accuracy, 4),
        "f1_scores": f1_scores,
        "n_training_samples": len(X_train),
        "algorithm": "LogisticRegression",
        "source": "bootstrap (iris dataset completo)",
        "activated": True
    }]
    save_history(history)
    print(f"[bootstrap] Modelo base creado → versión={version}, accuracy={accuracy:.4f}")


# ---------------------------------------------------------------------------
# App FastAPI
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Iris Continuous Training API",
    description=(
        "Extensión MLOps de app-iris. Sirve predicciones y permite reentrenar "
        "el modelo con nuevas muestras etiquetadas, registrando el historial de versiones."
    ),
    version="1.0.0",
)


@app.on_event("startup")
def startup_event():
    if not MODEL_PATH.exists():
        bootstrap_model()
    else:
        meta = get_active_model_meta()
        if meta:
            print(f"[startup] Modelo activo cargado → versión={meta['version']}, "
                  f"accuracy={meta['accuracy']}")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", tags=["Sistema"])
def health():
    meta = get_active_model_meta()
    return {
        "status": "ok",
        "active_model_version": meta["version"] if meta else "none",
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }


@app.post("/predict", response_model=PredictResponse, tags=["Inferencia"])
def predict(sample: IrisSample):
    """
    Realiza una predicción con el modelo activo.
    Devuelve la clase predicha, su nombre y la versión del modelo usado.
    """
    if not MODEL_PATH.exists():
        raise HTTPException(status_code=503, detail="Modelo no disponible. Llama primero a /train.")

    clf = joblib.load(MODEL_PATH)
    X = np.array([[
        sample.sepal_length,
        sample.sepal_width,
        sample.petal_length,
        sample.petal_width
    ]])
    pred = int(clf.predict(X)[0])
    meta = get_active_model_meta()

    return PredictResponse(
        prediction=pred,
        class_name=CLASS_NAMES[pred],
        model_version=meta["version"] if meta else "unknown"
    )


@app.post("/train", response_model=TrainResponse, tags=["Entrenamiento"])
def train(request: TrainRequest):
    """
    Reentrena el modelo con las nuevas muestras enviadas.

    - Si `retrain_from_scratch=False` (por defecto), las nuevas muestras se añaden
      al dataset de entrenamiento anterior (si existe) y se reentrena sobre el total.
    - Si `retrain_from_scratch=True`, solo se usan las muestras enviadas.
    - El nuevo modelo **reemplaza al activo solo si su accuracy ≥ accuracy anterior**.
    - Cada entrenamiento queda registrado en el historial aunque no se active.
    """

    # 1. Preparar nuevas muestras
    new_X = np.array([[s.sepal_length, s.sepal_width, s.petal_length, s.petal_width]
                       for s in request.samples])
    new_y = np.array([s.label for s in request.samples])

    # 2. Recuperar accuracy del modelo activo
    active_meta = get_active_model_meta()
    previous_accuracy = active_meta["accuracy"] if active_meta else None
    previous_f1 = active_meta.get("f1_scores", [0.0, 0.0, 0.0]) if active_meta else [0.0, 0.0, 0.0]

    # 3. Construir dataset de entrenamiento
    data_file = MODELS_DIR / "accumulated_data.joblib"

    if not request.retrain_from_scratch and data_file.exists():
        saved = joblib.load(data_file)
        X_train = np.vstack([saved["X"], new_X])
        y_train = np.concatenate([saved["y"], new_y])
        source = f"incremental (+{len(new_X)} muestras nuevas, {len(saved['X'])} anteriores)"
    else:
        X_train, y_train = new_X, new_y
        source = f"desde cero ({len(new_X)} muestras)"

    # 4. Necesitamos al menos 2 clases para entrenar
    if len(np.unique(y_train)) < 2:
        raise HTTPException(
            status_code=422,
            detail="El dataset de entrenamiento debe contener al menos 2 clases distintas."
        )

    # 5. Entrenar nuevo modelo
    clf_new = LogisticRegression(max_iter=300, random_state=42)

    # Evaluación: si hay suficientes datos, usamos split; si no, evaluamos en train
    if len(X_train) >= 20:
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42
        )
        clf_new.fit(X_tr, y_tr)
        y_pred_new = clf_new.predict(X_val)
        accuracy_new = float(accuracy_score(y_val, y_pred_new))
        f1_new = [round(float(f), 4) for f in f1_score(y_val, y_pred_new, average=None, labels=[0, 1, 2])]
        eval_note = f"validación con {len(X_val)} muestras"
    else:
        clf_new.fit(X_train, y_train)
        y_pred_new = clf_new.predict(X_train)
        accuracy_new = float(accuracy_score(y_train, y_pred_new))
        f1_new = [round(float(f), 4) for f in f1_score(y_train, y_pred_new, average=None, labels=[0, 1, 2])]
        eval_note = "evaluación en train (dataset pequeño, < 20 muestras)"

    accuracy_new = round(accuracy_new, 4)

    # 6. Decidir si activar el nuevo modelo
    model_updated = False
    message = ""

    if previous_accuracy is None:
        model_updated = True
        message = f"Nuevo modelo activado. Accuracy {accuracy_new:.4f} (primer modelo)"
    elif request.policy.method == "any_improvement":
        model_updated = accuracy_new >= previous_accuracy
        message = f"Policy any_improvement: Accuracy {accuracy_new:.4f} >= {previous_accuracy:.4f}"
    elif request.policy.method == "min_delta":
        model_updated = accuracy_new >= (previous_accuracy + request.policy.threshold)
        message = f"Policy min_delta: Mejora {accuracy_new - previous_accuracy:.4f} >= {request.policy.threshold}"
    elif request.policy.method == "per_class_f1":
        idx = request.policy.target_class
        model_updated = f1_new[idx] > previous_f1[idx]
        message = f"Policy per_class_f1 (clase {idx}): {f1_new[idx]:.4f} > {previous_f1[idx]:.4f}"

    version = f"v{len(load_history()) + 1}.0-{uuid.uuid4().hex[:6]}"
    status = "activado" if model_updated else "rechazado"

    if model_updated:
        joblib.dump(clf_new, MODEL_PATH)
        joblib.dump({"X": X_train, "y": y_train}, data_file)
    else:
        message = f"Modelo NO activado. {message}. El modelo activo se mantiene sin cambios."

    # 7. Registrar en historial
    history = load_history()
    history.append({
        "version": version,
        "trained_at": datetime.utcnow().isoformat() + "Z",
        "accuracy": accuracy_new,
        "f1_scores": f1_new,
        "n_training_samples": len(X_train),
        "algorithm": "LogisticRegression",
        "source": source,
        "eval_note": eval_note,
        "status": status,
        "activated": model_updated,
        "policy_used": request.policy.method,
        "reason": message
    })
    save_history(history)

    return TrainResponse(
        status=status,
        model_version=version,
        accuracy_new=accuracy_new,
        accuracy_previous=previous_accuracy,
        model_updated=model_updated,
        message=message
    )


@app.get("/model/info", response_model=ModelInfo, tags=["Modelo"])
def model_info():
    """
    Devuelve información del modelo activo y el historial completo de entrenamientos.
    """
    history = load_history()
    if not history:
        raise HTTPException(status_code=404, detail="No hay ningún modelo entrenado aún.")

    active = get_active_model_meta()

    return ModelInfo(
        active_version=active["version"],
        trained_at=active["trained_at"],
        accuracy=active["accuracy"],
        n_training_samples=active["n_training_samples"],
        algorithm=active["algorithm"],
        history=history
    )


@app.delete("/model/history", tags=["Modelo"])
def reset_history():
    """
    [CUIDADO] Elimina el historial y el modelo activo. Fuerza bootstrap en el próximo arranque.
    Útil para pruebas y demostración.
    """
    if HISTORY_PATH.exists():
        HISTORY_PATH.unlink()
    if MODEL_PATH.exists():
        MODEL_PATH.unlink()
    data_file = MODELS_DIR / "accumulated_data.joblib"
    if data_file.exists():
        data_file.unlink()
    bootstrap_model()
    return {"status": "ok", "message": "Historial eliminado y modelo base restaurado."}