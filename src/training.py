import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib


def train_model(data_path="../data/processed", model_path="../models", train_path="../data/train", test_path="../data/test"):
    # Cargar datos procesados
    housing_df = pd.read_csv(data_path)

    # Definir características y objetivo
    features = ['bedrooms', 'bathrooms', 'sqft_living_log', 'sqft_lot', 'floors', 
                'view', 'grade', 'sqft_above', 'sqft_basement', 
                'sqft_living15', 'sqft_lot15', 'property_age']
    X = housing_df[features]
    y = housing_df['price_log']

    # Dividir en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Guardar los conjuntos de datos
    X_train.to_csv(f"{train_path}/X_train.csv", index=False)
    y_train.to_csv(f"{train_path}/y_train.csv", index=False)
    X_test.to_csv(f"{test_path}/X_test.csv", index=False)
    y_test.to_csv(f"{test_path}/y_test.csv", index=False)

    # Entrenar el modelo
    model = RandomForestRegressor(n_estimators=200, max_depth=20, random_state=42)
    model.fit(X_train, y_train)

    # Guardar el modelo
  
    joblib.dump(model, f"{model_path}/final_model.pkl")
    print("Modelo entrenado y guardado en", model_path)

