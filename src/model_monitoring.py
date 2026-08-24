#src/model_monitoring.py

#librerías
import pandas as pd
import os
#Librería para crear la aplicación  en Streamlit
import streamlit as st
#librerías de visualización
import matplotlib.pyplot as plt
#SKlearn para procesamiento de datos
from sklearn.model_selection import train_test_split
#importar el método cargar datos
from cargar_datos  import cargarDatos

#########
#1. Configuración de la aplicación
#########

API_URL = "http://localhost:8000/predict" #URL de la API FastAPI
DATASET_PATH = "..." #Ruta del dataset "simulado"
MONITOR_LOG = "./Base_de_datos.xlsx" #Ruta del archivo del log de monitorización

################
###2. Cargar el dataset y dividir los datos
######

@st.cache_data
def load_data():
    #2.1 Llamamos a la función cargar_datos() para obtener el dataframe
    df = cargarDatos()

    print(df)

    #2.2 Creamos los features y el target

    target = "Pago_atiempo"
    X = df.drop(columns=[target])   # features
    y = df[target]                  # target

    #2.3 División de datos de prueba y entrenamiento
    X_ref, X_new, y_ref, y_new = train_test_split(X, y, test_size=0.2,
                                                   random_state=42, stratify = y)

X_ref, X_new, y_ref, y_new = load_data()

#########
#3. Crear interfaz inicial
########

st.title("-Aplicación para el monitoreo de los datos-")