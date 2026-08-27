# comando para ejecutar la API: uvicorn model_deploy:app --reload
import pandas as pd
import joblib
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

#Inicialización de la API
####

app = FastAPI(title = "API de predicción de pago a tiempo",
                description = """Esta API permite predecir si un cliente pagará a tiempo o no, 
                utilizando un modelo de machine learning previamente entrenado.""",
                version = "1.1.1",

)

#Cargar modelo entrenado
MODEL_PATH = "models/xgb_model.joblib"
try:
    with open(MODEL_PATH, "rb") as f:
        model = joblib.load(f)

    print("Modelo cargado exitosamente")
except Exception as e:
    print(f"Error al cargar el modelo: {e}")
    model = None

#Definir endpoint de la API

@app.get("/saludo")
def saludo():
    return {"message": """¡Hola! Esta APIsirve para predecir si un cliente pagará a tiempo o no.
                             Además, estoy corriendo desde el contenedor de Docker"""}


#Endpoint de saludo
@app.post("/predict")
def predict_batch(input_data: dict):
    if model is None:
        return {"El modelo no pudo ser cargado. Revisa los logs del servidor para más detalles"}

        try:
            return {"El modelo está cargado y listo para hacer predicciones"}

        except Exception as e:
            return {"Error al hacer la predicción": str(e)}