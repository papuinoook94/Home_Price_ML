# Importar librerías
import pandas as pd
import numpy as np



def process_data(input_path="../data/raw", output_path="../data/processed"):
    # Cargar datos crudos
    input_path = '../data/raw/Housing.csv'
    housing_df = pd.read_csv(input_path)
    # Convertir la columna 'date' a den formato datetime
    housing_df['date'] = pd.to_datetime(housing_df['date'], errors='coerce')
    # Recalcular la correlacion_matrix sin 'date'
    housing_df = housing_df.drop(columns=['date']).corr(numeric_only=True)
    # Aplicar transformaciones: crear 'price_log' y 'sqft_living_log'
    housing_df['price_log'] = np.log1p(housing_df['price'])
    housing_df['sqft_living_log'] = np.log1p(housing_df['sqft_living'])
    # Crear una nueva característica 'property_age'
    housing_df['property_age'] = 2024 - housing_df['yr_built']

    # Eliminar valores atípicos extremos en 'sqft_lot'
    sqft_lot_threshold = housing_df['sqft_lot'].quantile(0.99)
    housing_df = housing_df[housing_df['sqft_lot'] <= sqft_lot_threshold]

    selected_features = ['sqft_living', 'grade', 'bathrooms', 'distance_to_center', 'property_age', 'bedrooms','view','lat','floors','waterfront']
    housing_df = housing_df[selected_features]

    # Guardar datos procesados en la carpeta data/processed
    housing_df.to_csv(f"{output_path}/Housing_Processed.csv", index=False)
    print("Datos procesados y guardados en", output_path)


  
  
