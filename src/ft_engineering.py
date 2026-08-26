# Importar librerías
import pandas as pd
import numpy as np
import seaborn as sns

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score

from Cargar_datos import cargarDatos

import warnings
warnings.filterwarnings('ignore')

def clean_data(df):
    """
    Función para para limpieza de datos corruptos
    """
    #Copia para no afectar el dataset original
    df_clean = df.copy()

    valid_values = ["Estable", "Creciente", "Decreciente"]
    df_clean["tendencia_ingresos"] = df_clean["tendencia_ingresos"].apply(
        lambda x: x if x in valid_values else None
    )


def split_data(df):
    """
    Separación de datos de entrenamiento
    """    
    #Revisar si sacar la variable de fecha
    target = "Pago_atiempo"
    X = df.drop(columns=[target])   # features
    y = df[target]                  # target

    #Tener en cuenta la tenporalidad. Shuffle = False
    X_train, y_train, X_test, y_test = train_test_split(X, y, test_size=20, shuffle=False)

    return X_train, y_train, X_test, y_test


#Pipelines, falta el standaaaaarr scaleeerrrrrrrrrMIRALOOO
def preprocesing_pipeline(numerical_features, nominal_features, ordinal_features):
    """
    Construye y retorna la estructura del ColumnTransformer aplicando
    OneHotEncoder para nominales y OrdinalEncoder para ordinales
    """
    #pipeline variables numéricas
    numerical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ])


    #pipeline variables categóricas
    categorical_pipeline = Pipeline(
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("One_hot", OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=False))# Para que devuelva matriz densa tradicional de NumPy
    )

    order_trend = ["Decreciente", "Estable", "Creciente"]
    ordinal_cat_pipeline = Pipeline(
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ordinal", OrdinalEncoder(categories=[order_trend]))
    )

    #Integración
    preprocessor = ColumnTransformer(transformers=[
        ("num", numerical_pipeline, numerical_features),
        ("cat", categorical_pipeline, nominal_features),
        ("cat_ord", ordinal_cat_pipeline,ordinal_features)
    ])
    return preprocessor



def ft_engineering_pipeline(df):
    """
    Función orquestadora que ejecuta todo el proceso de ingeniería de características
    Distingue variables numéricas, nominales y ordinales
    """
    #limpieza con la función clean_data
    df = clean_data(df)

    #Separamos features y target
    target = "Pago_atiempo"
    #X_temp = df.drop(columns=[target, "fecha_prestamo"], errors="ignore")
    X = df.drop(columns=[target])   # features
    y = df[target]                  # target


    #Separación de features
    numerical_features = X.select_dtypes(include=['int64','float64']).columns.tolist()
    #categorical_features = X.select_dtypes(include=['object']).columns.tolist()
    nominal_features = ["tipo_laboral"]
    ordinal_features = ["tendencia_ingresos"]

    print(f"Variables numéricas: {numerical_features}")
    print(f"Variables categóricas: {nominal_features}")
    print(f"Variables categóricas ordinales: {ordinal_features}")

    #Separacion de data
    X_train, y_train, X_test, y_test = split_data(df)

    #Llamar a Pipeline preprocesamiento
    preprocessor = preprocesing_pipeline(numerical_features, nominal_features, ordinal_features)

    #Llamar Entrenamiento de modelos, tener en cuenta la tenporalidad. Shuffle = False
    #preprocesador aprende fit sólo de entrenamiento para evitar data lekeage
    preprocessor.fit(X_train)
    X_train_processed = preprocessor.transform(X_train)
    X_test_processed = preprocessor.transform(X_train)

    print("Ingeniería de Características Exitos (Con Encoders Corregidos) ---")
    print(f"Numéricas: {len(numerical_features)} | Nominales: {len(nominal_features)} | Ordinales: {len(ordinal_features)}")
    print(f"Dimensiones Finales - X_train: {X_train_processed.shape} | X_test: {X_test_processed.shape}")

    return X_train_processed, X_test_processed


def main():
    print("Iniciando Pipeline de Ingeniería de Características desde MAIN")
    df=cargarDatos()
    print(df)

    #Ejecutar el pipeline inserter el codigo AQUIII
    X_train, X_test, y_train, y_test, preprocessor = ft_engineering_pipeline (df)

    return X_train, X_test, y_train, y_test

#Para ejecución directa en terminal
if __name__ == "__main__":
    main()