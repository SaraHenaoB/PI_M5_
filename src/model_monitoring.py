#src/model_monitoring.py
# run this command: streamlit run model_monitoring.py
#librerías
import pandas as pd
import os
import datetime
import streamlit as st                                      #Librería para crear la aplicación  en Streamlit
import requests
import matplotlib.pyplot as plt                             #librerías de visualización
from scipy.stats import ks_2samp                            #Kolmovorov-Smirnov variables numericas
from scipy.stats import chi2_contingency                    #chi chuadrado variable categoricas
from sklearn.model_selection import train_test_split        #SKlearn para procesamiento de datos

from Cargar_datos  import cargarDatos                       #importar el método cargar datos

#########
# Configuración de la aplicación

API_URL = "http://localhost:8000/predict" #URL de la API FastAPI
MONITOR_LOG = "./Base_de_datos.xlsx" #Ruta del archivo del log de monitorización



@st.cache_data    #guarda el resultado de esa función en la memoria caché para no  calcularla desde cero cada vez que la página se actualice

def load_and_split_monitoring_data():
    """
    Carga el dataset y lo divide en Referencia (80%) y Producción (20%).
    Simula Data Drift alterando variables financieras clave.
    """
    df = cargarDatos()
    target = "Pago_atiempo"
    
    # Ordenamiento cronológico para simular producción real
    if "fecha_prestamo" in df.columns:
        df["fecha_prestamo"] = pd.to_datetime(df["fecha_prestamo"], format="%m/%d/%Y %H:%M", errors="coerce")
        df = df.dropna(subset=["fecha_prestamo"]).sort_values(by="fecha_prestamo").reset_index(drop=True)
    
    # Purgar variables soplones (Data Leakage)
    fuga_features = ['saldo_mora', 'saldo_mora_codeudor', 'saldo_total', 'saldo_principal', 'puntaje']
    df = df.drop(columns=[col for col in fuga_features if col in df.columns], errors="ignore")
    
    X = df.drop(columns=[target, "fecha_prestamo"], errors="ignore")
    y = df[target]
    
    # 80% antiguo es Referencia (X_ref) y 20% reciente es Producción (X_new)
    X_ref, X_new, y_ref, y_new = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    # Simulación de Data Drift (Crisis económica)
    X_new_drifted = X_new.copy()
    if "salario_cliente" in X_new_drifted.columns:
        X_new_drifted["salario_cliente"] = X_new_drifted["salario_cliente"] * 0.7
    if "capital_prestado" in X_new_drifted.columns:
        X_new_drifted["capital_prestado"] = X_new_drifted["capital_prestado"] * 1.4
        
    return X_ref, X_new_drifted




def calculate_data_drift(X_ref, X_new):
    """
    Pruebas estadísticas de Kolmogorov-Smirnov y Chi-cuadrado 
    con blindaje de tipos de datos para evitar caídas en Pandas.
    """
    drift_results = []
    
    numerical_features = X_ref.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X_ref.select_dtypes(include=['object', 'str', 'category']).columns.tolist()
    
    #Variables Numéricas (KS-Test)
    for var in numerical_features:
        ref_data = X_ref[var].dropna()
        new_data = X_new[var].dropna()
        
        if len(ref_data) > 0 and len(new_data) > 0:
            stat, p_val = ks_2samp(ref_data, new_data)
            drift_detected = "Sí" if p_val < 0.05 else "No"
            
            drift_results.append({
                "Variable": var,
                "Tipo": "Numérica",
                "Métrica/Stat": round(stat, 4),
                "P-Value": round(p_val, 5),
                "Drift_Detected": drift_detected
            })
            
    #Variables Categóricas (Chi-Square con corrección de tipos en índices)
    for var in categorical_features:
        ref_counts = X_ref[var].value_counts()
        new_counts = X_new[var].value_counts()
        
        # Convertir índices a string para evitar colisiones int vs str
        ref_counts.index = ref_counts.index.astype(str)
        new_counts.index = new_counts.index.astype(str)
        
        contingency_df = pd.DataFrame({'Ref': ref_counts, 'New': new_counts}).fillna(0)
        
        if contingency_df.shape[0] > 1: 
            try:
                stat, p_val, _, _ = chi2_contingency(contingency_df)
                drift_detected = "Sí" if p_val < 0.05 else "No"
                
                drift_results.append({
                    "Variable": var,
                    "Tipo": "Categórica",
                    "Métrica/Stat": round(stat, 4),
                    "P-Value": round(p_val, 5),
                    "Drift_Detected": drift_detected
                })
            except:
                pass 
                
    return pd.DataFrame(drift_results)

def save_prediction_log(new_client_data, prediction, probability):
    """
    Persiste en un archivo Excel histórico los datos de entrada junto con 
    los pronósticos entregados por el modelo para auditoría periódica.
    """
    log_entry = {**new_client_data}
    log_entry["Query_Timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry["Model_Prediction"] = int(prediction)
    log_entry["Probability_Pago_A_Tiempo"] = round(float(probability), 4)
    
    df_new_row = pd.DataFrame([log_entry])
    
    if os.path.exists(MONITOR_LOG):
        try:
            df_historical = pd.read_excel(MONITOR_LOG)
            df_updated = pd.concat([df_historical, df_new_row], ignore_index=True)
            df_updated.to_excel(MONITOR_LOG, index=False)
        except Exception as e:
            st.error(f"Error al guardar en registro histórico Excel: {e}")
    else:
        df_new_row.to_excel(MONITOR_LOG, index=False)

def main():
    st.set_page_config(page_title="Dashboard de Monitoreo MLOps", layout="wide")
    
    # Menú lateral para alternar las vistas solicitadas en la consigna
    app_mode = st.sidebar.selectbox(
        "Seleccione el Módulo de la App:", 
        ["Dashboard de Monitoreo", "Evaluación de Crédito en Tiempo Real"]
    )
    
    try:
        X_ref, X_new = load_and_split_monitoring_data()
    except Exception as e:
        st.error(f"Error al cargar los datos base: {e}")
        return

    # --- ENTORNO 1: MONITOREO DE DATA DRIFT ---
    if app_mode == "Dashboard de Monitoreo":
        st.title("📊 Aplicación para el Monitoreo de los Datos (Data Drift)")
        st.markdown("### Estado de Estabilidad del Sistema")
        
        df_drift = calculate_data_drift(X_ref, X_new)
        
        total_vars = len(df_drift)
        drifted_vars = len(df_drift[df_drift["Drift_Detected"] == "Sí"])
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Variables Monitoreadas", total_vars)
        col2.metric("Variables con Drift", drifted_vars, delta=f"{drifted_vars} críticas", delta_color="inverse")
        
        if drifted_vars > 2:
            col3.metric("Estado del Sistema", "⚠️ RE-ENTRENAR", delta="Acción requerida", delta_color="off")
        else:
            col3.metric("Estado del Sistema", "✅ ESTABLE", delta="Sin anomalías", delta_color="normal")
            
        st.subheader("📋 Matriz de Diagnóstico Poblacional")
        st.dataframe(df_drift, use_container_width=True)
        
    # --- ENTORNO 2: INFERENCIA LOGUEADA EN TIEMPO REAL ---
    elif app_mode == "Evaluación de Crédito en Tiempo Real":
        st.title("🌟 Evaluación de Crédito en Tiempo Real")
        st.write(f"Conectado al endpoint de FastAPI en: `{API_URL}`")
        st.write(f"Historial de auditoría configurado en: `{MONITOR_LOG}`")
        st.divider()
        
        # Formulario estructurado en tres pestañas limpias
        tab1, tab2, tab3 = st.tabs(["💰 Datos Financieros", "📄 Detalles del Crédito", "👤 Perfil Ocupacional"])
        
        with tab1:
            st.subheader("Ingresos del Solicitante")
            salario_cliente = st.number_input("Salario Mensual Cliente ($)", min_value=0.0, value=3500000.0, step=50000.0)
            
            no_datacredito = st.checkbox("¿El cliente no tiene histórico registrado en la central (DataCrédito)?")
            promedio_ingresos_datacredito = None if no_datacredito else st.number_input(
                "Promedio Ingresos DataCrédito ($)", min_value=0.0, value=2800000.0, step=50000.0
            )

        with tab2:
            st.subheader("Condiciones del Préstamo Solicitado")
            capital_prestado = st.number_input("Monto del Crédito Solicitado ($)", min_value=0.0, value=10000000.0, step=100000.0)
            plazo_meses = st.number_input("Plazo del Crédito (Meses)", min_value=1, max_value=120, value=36, step=1)
            tipo_credito = st.selectbox("Línea de Crédito", options=["Libre Inversión", "Microcrédito", "Vivienda", "Vehículo"])

        with tab3:
            st.subheader("Soporte Laboral")
            tipo_laboral = st.selectbox("Tipo de Vinculación Laboral", options=["Asalariado", "Independiente", "Pensionado", "Informal"])
            tendencia_ingresos = st.selectbox(
                "Tendencia de Ingresos Evaluada", 
                options=["sin información", "Decreciente", "Estable", "Creciente"],
                index=2
            )
            
        client_payload = {
            "salario_cliente": float(salario_cliente),
            "promedio_ingresos_datacredito": promedio_ingresos_datacredito,
            "capital_prestado": float(capital_prestado),
            "plazo_meses": int(plazo_meses),
            "tipo_credito": str(tipo_credito),
            "tipo_laboral": str(tipo_laboral),
            "tendencia_ingresos": str(tendencia_ingresos)
        }
        
        st.divider()
        
        if st.button("🚀 Evaluar Solicitud de Crédito", use_container_width=True):
            with st.spinner("Enviando parámetros al modelo XGBoost de producción..."):
                try:
                    response = requests.post(API_URL, json=client_payload, timeout=5)
                    
                    if response.status_code == 200:
                        res_data = response.json()
                        prediction = res_data["prediction"]
                        probability = res_data["probability"]
                        
                        if prediction == 1:
                            st.success(f"✅ **CRÉDITO APROBADO**: Confianza de pago a tiempo del {probability * 100:.2f}%")
                        else:
                            st.error(f"⚠️ **CRÉDITO RECHAZADO**: Alto riesgo de impago detectable. Probabilidad de éxito de solo {probability * 100:.2f}%")
                            
                        # Persiste de forma obligatoria los datos y el pronóstico entregado en el log Excel
                        save_prediction_log(client_payload, prediction, probability)
                        st.toast("Inferencia registrada con éxito en logs de auditoría.", icon="💾")
                    else:
                        st.error(f"Error de Inferencia Backend ({response.status_code}): {response.text}")
                        
                except requests.exceptions.ConnectionError:
                    st.error(f"Fallo de infraestructura: Imposible conectar con la API en '{API_URL}'. Verifica que FastAPI esté arriba.")
                except Exception as e:
                    st.error(f"Ocurrió una anomalía inesperada al procesar el JSON: {e}")

if __name__ == "__main__":
    main()