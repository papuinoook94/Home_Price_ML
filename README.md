# Predicción de Precios de Propiedades

Este proyecto predice el precio de una propiedad en función de sus características. Utiliza modelos de machine learning entrenados con datos procesados de propiedades provenientes de Kaggle [https://www.kaggle.com/datasets/sukhmandeepsinghbrar/housing-price-dataset/data] y una aplicación web en **Streamlit** para facilitar la interacción.

## Descripción de variables de dataset.
- id: Identificador único de la propiedad.
- date: Fecha de listado de la propiedad.
- price: Precio de la propiedad (variable objetivo).
- bedrooms: Número de habitaciones.
- bathrooms: Número de baños.
- sqft_living: Área habitable en pies cuadrados.
- sqft_lot: Tamaño total del lote en pies cuadrados.
- floors: Número de pisos.
- waterfront: Vista al agua (1 = sí, 0 = no).
- view: Calidad de la vista (0 a 4).
- condition: Estado general de la propiedad (1 a 5).
- grade: Calidad de construcción y diseño (1 a 13).
- sqft_above: Área sobre el nivel del suelo en pies cuadrados.
- sqft_basement: Área del sótano en pies cuadrados.
- yr_built: Año de construcción.
- yr_renovated: Año de la última renovación.
- zipcode: Código postal de la ubicación.
- lat: Coordenadas geográficas de latitud de la propiedad.
- long: Coordenadas geográficas de longitud la propiedad.
- sqft_living15: Promedio del área habitable de las 15 propiedades más cercanas.
- sqft_lot15: Promedio del tamaño del lote de las 15 propiedades más cercanas

## Estructura del Proyecto

La estructura del proyecto es la siguiente:
```
project/
│
├── data/                   # Contiene los datos del proyecto
│   ├── raw/                # Datos crudos sin procesar
│   ├── processed/          # Datos procesados y transformados
│   ├── train/              # Conjuntos de datos de entrenamiento
│   └── test/               # Conjuntos de datos de prueba
│
├── models/                 # Almacena los modelos entrenados y la configuración
│   ├── final_model.pkl     # Modelo final guardado en formato pickle
│   └── model_config.yaml   # Configuración de los parámetros del modelo
│
├── app_streamlit/          # Aplicación en Streamlit
│   ├── app.py              # Código principal de la aplicación
│   └── requirements.txt    # Dependencias de la aplicación
│
├── notebooks/              # Notebooks de Jupyter para cada etapa del análisis
│   ├── 01_Fuentes.ipynb    # Adquisición de datos
│   ├── 02_LimpiezaEDA.ipynb # Análisis exploratorio y limpieza de datos
│   └── 03_Entrenamiento_Evaluacion.ipynb # Entrenamiento y evaluación de modelos
│
├── src/                    # Código fuente del proyecto
│   ├── data_processing.py  # Procesamiento y transformación de datos
│   ├── training.py         # Entrenamiento del modelo
│   └── evaluation.py       # Evaluación del modelo
│
└── README.md               # Descripción del proyecto

```
## Requisitos

### Dependencias
Instala las dependencias necesarias para ejecutar la aplicación y los scripts en `requirements.txt` dentro de la carpeta `app_streamlit`:

```bash
pip install -r app_streamlit/requirements.txt
```


## Estructura de Archivos
Datos: Coloca los datos crudos en data/raw/.
Modelos: Los modelos entrenados se guardarán en la carpeta models/.
Aplicación Web: Ejecuta app.py desde app_streamlit para interactuar con el modelo final.

## Ejecución de Scripts
### Procesamiento de Datos
Para limpiar y transformar los datos:

```bash
python src/data_processing.py
```
Esto tomará los datos en data/raw/, los procesará y guardará en data/processed/.

### Entrenamiento del Modelo
Para entrenar el modelo:

```bash
python src/training.py
```
Este script entrena el modelo y guarda los conjuntos de entrenamiento y prueba en data/train/ y data/test/.

### Evaluación del Modelo
Para evaluar el modelo:

```bash
python src/evaluation.py
El script evalúa el modelo utilizando el conjunto de prueba y muestra métricas de rendimiento como el Mean Squared Error (MSE) y R-squared (R²).
```
## Aplicación Web
La aplicación en Streamlit permite al usuario ingresar las características de una propiedad y obtener una predicción del precio.

Para ejecutar la aplicación:

Navega a la carpeta app_streamlit.

Ejecuta:

```bash
streamlit run app.py
```
Esto abrirá la aplicación en el navegador, permitiendo interactuar con el modelo y obtener predicciones.

## Notebooks
Cada notebook documenta una etapa específica del proyecto:

- 01_Fuentes.ipynb: Carga y exploración inicial de datos.
- 02_LimpiezaEDA.ipynb: Limpieza de datos, análisis exploratorio y creación de nuevas características.
- 03_Entrenamiento_Evaluacion.ipynb: Entrenamiento, evaluación y optimización de modelos.

### Contacto
Si tienes preguntas o deseas colaborar, puedes contactarme en pabloprm94@gmail.com