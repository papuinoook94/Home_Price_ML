import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def evaluate_model(model_path="../models", test_path="../data/test/"):
    # Cargar el modelo
    model = joblib.load(f"{model_path}/final_model.pkl")

    # Cargar datos de prueba
    X_test = pd.read_csv(f"{test_path}/X_test.csv")
    y_test = pd.read_csv(f"{test_path}/y_test.csv").squeeze()  # .squeeze() para convertirlo en una serie

    # Realizar predicciones
    y_pred = model.predict(X_test)

    # Calcular métricas de evaluación
    rmse = np.sqrt(mean_squared_error)(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("Evaluación del modelo:")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")
    print(f"R2: {r2:.2f}")

# Ruta de ejemplo para ejecutar
# evaluate_model('../models', '../data/test')
