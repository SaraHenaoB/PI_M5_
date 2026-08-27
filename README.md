# PI_M5_
**Nota importante:** El avance 4 presenta erroes ya que hubo conflictos al inicializar docker aún habiendo hecho pruebas en 2 computadores distintos
por lo que no se pudo resolver el contenerizado de la aplicación

Proyecto Integrador: Sistema de Monitoreo de Datos e Inferencia de Riesgo Crediticio

## contexto

Has iniciado tu labor en el equipo de Datos y Analítica de una empresa financiera, desempeñándote como Científico de Datos Junior Advanced. Tu primera asignación consiste en desarrollar un modelo predictivo mediante técnicas de aprendizaje automático, utilizando información histórica de créditos, con el objetivo de anticipar el comportamiento de nuevos usuarios.
La empresa opera bajo un esquema estructurado de proyectos, en el cual cada iniciativa debe seguir una arquitectura de carpetas estrictamente definida. Esta estructura no puede ser modificada, ya que los procesos de despliegue a producción están automatizados a través de pipelines de validación en Jenkins. Cualquier alteración en la organización de carpetas podría generar retrasos significativos en el paso a producción.

El sistema está estructurado de manera modular siguiendo las mejores prácticas de **MLOps**.

---

## Objetivo del Negocio

El sistema predice la probabilidad de que un cliente pague un crédito financiero a tiempo o entre en mora:
*   **Variable Objetivo ('Pago_atiempo')**: 
    *   1: El cliente paga a tiempo.
    *   0: El cliente entra en mora (Clase Crítica).
*   **Desbalanceo Crítico**: 95% pertenecen a la clase 1 y solo 5% a la clase 0. El pipeline tiene en cuenta este desbalanceo para el entrenamiento de modelos.

---

## Arquitectura del Software y Componentes

El proyecto está modularizado en scripts de Python independientes para asegurar mantenibilidad y escalabilidad en entornos productivos:

### 1. Cargar_datos.py
Contiene la función `cargarDatos()` automatizada para la ingesta del dataset crudo de 10,763 registros y 23 columnas.

### 2. `ft_engineering.py`
*   **Mitigación de Fuga de Datos (Data Leakage)**: Purga automática de variables posteriores al otorgamiento del crédito (`saldo_mora`, `saldo_mora_codeudor`, `saldo_total`, `saldo_principal`, `puntaje`).
*   **Validación Temporal**: Ordenamiento cronológico por `fecha_prestamo` y división estricta de datos (80% Entrenamiento / 20% Prueba) con `shuffle=False` para evitar fuga temporal.
*   **Pipeline Unificado (`ColumnTransformer`)**:
    *   *Numéricas*: Imputación por mediana con parámetro `add_indicator=True` para rastrear de forma transparente el 27% de valores nulos en `promedio_ingresos_datacredito`, seguido de un escalamiento estándar (`StandardScaler`).
    *   *Categorías Nominales*: Imputación por moda (`most_frequent`) + `OneHotEncoder(drop="first", sparse_output=False)` en `tipo_laboral`.
    *   *Categorías Ordinales*: Tratamiento de nulos como categoría explícita ("sin información") + `OrdinalEncoder` + `StandardScaler` para `tendencia_ingresos`.

### 3. `model_monitoring.py` (Módulo Central - Avance 3)
Este componente unifica el motor matemático de detección de desvíos y el frontend de producción mediante **Streamlit**. Se divide en dos entornos independientes controlados por la barra lateral:

#### Entorno 1: Dashboard de Monitoreo (Data Drift)
Ejecuta de forma automatizada pruebas estadísticas para comparar la población de referencia (80% histórico) contra la población de producción simulada (20% reciente modificada con una reducción salarial del 30% e incremento de capital solicitado del 40%):
*   **Variables Numéricas**: Prueba de dos muestras de **Kolmogorov-Smirnov (`ks_2samp`)**.
*   **Variables Categóricas**: Prueba de independencia de **Chi-cuadrado (`chi2_contingency`)**.
*   **Umbral de Alerta**: Un p-valor $< 0.05$ marca la variable con `Drift_Detected: Sí`. Si más de 2 variables registran desviación, el sistema dispara automáticamente una **Alerta de Re-entrenamiento**.
*   *Robustez MLOps*: Los índices de las frecuencias categóricas se convierten forzosamente a cadenas de texto (`index.astype(str)`) antes de ensamblar la matriz de contingencia de Pandas, protegiendo al sistema contra colisiones de tipos (`int` vs `str`).

#### Entorno 2: Evaluación de Crédito en Tiempo Real
*   **Captura de Datos**: Formulario estructurado y optimizado con pestañas limpias (`st.tabs`) para organizar las variables del perfil financiero, detalles del crédito y soporte laboral.
*   **Inferencia HTTP**: Orquestado para empaquetar los campos y despachar un payload JSON usando la librería `requests` hacia la URL del backend (`http://localhost:8000/predict`).
*   **Registro de Inferencia (Auditoría Histórica)**: Al presionar el botón de evaluación, la función `save_prediction_log()` intercepta los datos de entrada junto con la predicción del modelo y la estampa de tiempo (`Timestamp`), persistiendo los registros de forma obligatoria en la bitácora local en formato Excel: `./Base_de_datos.xlsx`.

---

## Instrucciones de Ejecución Local

1.  **Instalar Dependencias**: Garantizar que las librerías base estén instaladas en su entorno:
    ```bash
    pip install -r requirements.txt
    ```
2.  **Iniciar la Aplicación**: Ejecutar el servidor web local de Streamlit desde la raíz del proyecto:
    ```bash
    streamlit run src/model_monitoring.py
    ```

