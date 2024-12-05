import streamlit as st
import pandas as pd
import numpy as np
import joblib
import yaml

# Cargar el modelo final
model = joblib.load('final_model.pkl')

# Cargar configuración del modelo (RMSE y MAE, si es necesario)
with open('model_config.yaml', 'r') as file:
    model_config = yaml.safe_load(file)

# Métricas del modelo (previamente calculadas durante el entrenamiento)
rmse = model_config.get('rmse', 50000)  # RMSE estimado en dólares
mae = model_config.get('mae', 30000)   # MAE estimado en dólares

# Título de la aplicación
st.title("Predicción del Rango de Precios de Propiedades")

# Instrucciones
st.write("Introduce las características de la propiedad para obtener un rango estimado del precio basado en las métricas del modelo (RMSE y MAE).")

# Variables seleccionadas
selected_features = ['sqft_living', 'grade', 'bathrooms', 'distance_to_center', 
                     'property_age', 'bedrooms', 'view', 'lat', 'floors', 'waterfront']

# Entradas de usuario para cada característica
sqft_living = st.number_input("Área habitable en pies cuadrados", min_value=500, max_value=10000, value=2000)
grade = st.slider("Calidad de construcción (1 a 13)", min_value=1, max_value=13, value=7)
bathrooms = st.number_input("Número de baños", min_value=1, max_value=10, value=2)
distance_to_center = st.number_input("Distancia al centro de Seattle (millas)", min_value=0, max_value=15, value=7)
property_age = st.number_input("Edad de la propiedad (años)", min_value=0, max_value=150, value=50)
bedrooms = st.number_input("Número de habitaciones", min_value=1, max_value=10, value=3)
view = st.selectbox("Calidad de la vista (0 a 4)", options=[0, 1, 2, 3, 4])
lat = st.number_input("Latitud", min_value=47.0, max_value=47.8, value=47.5)
floors = st.number_input("Número de pisos", min_value=1, max_value=3, value=1)
waterfront = st.selectbox("¿Vista al agua?", options=[0, 1], format_func=lambda x: "Sí" if x == 1 else "No")

# Crear DataFrame con las variables seleccionadas
input_data = pd.DataFrame({
    'sqft_living': [sqft_living],
    'grade': [grade],
    'bathrooms': [bathrooms],
    'distance_to_center': [distance_to_center],
    'property_age': [property_age],
    'bedrooms': [bedrooms],
    'view': [view],
    'lat': [lat],
    'floors': [floors],
    'waterfront': [waterfront]
})

# Realizar predicción y calcular rango
if st.button("Predecir Rango de Precio"):
    price_pred = model.predict(input_data)[0]  # Predicción del precio base
    
    # Calcular rango de precios
    lower_bound = price_pred - rmse - mae
    upper_bound = price_pred + rmse + mae
    
    # Mostrar el rango de precios
    st.write(f"El precio estimado de la propiedad está en el rango de: **${lower_bound:,.2f} - ${upper_bound:,.2f}**")
    st.write(f"Precio base calculado: ${price_pred:,.2f}")
    
