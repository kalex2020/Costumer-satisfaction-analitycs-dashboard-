# -*- coding: utf-8 -*-
# ======================================================================
# SCRIPT: ENTRENAMIENTO REGRESIÓN LINEAL
# Problema: Predicción de tiempo_real_min en logística última milla
# ======================================================================

import os
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge, Lasso, ElasticNet, LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import joblib
from pathlib import Path

# ======================================================================
# 1. CONFIGURACIÓN DE RUTAS
# ======================================================================

BASE_PATH = r"K:\Dashboard Delivery"
DATA_PATH = os.path.join(BASE_PATH, "data", "processed", "Train", "linear")
VAL_PATH = os.path.join(BASE_PATH, "data", "processed", "validation", "linear")
MODEL_PATH = os.path.join(BASE_PATH, "models", "linear_model.joblib")
METRICS_CSV_PATH = os.path.join(BASE_PATH, "trainig_models", "metrics", "model_metrics.csv")

# Crear directorio de métricas si no existe
os.makedirs(os.path.dirname(METRICS_CSV_PATH), exist_ok=True)

# ======================================================================
# 2. CARGA DE DATOS
# ======================================================================

print("=" * 70)
print("ENTRENAMIENTO: REGRESIÓN LINEAL")
print("=" * 70)

print("\n[1] Cargando datos de entrenamiento...")
try:
    X_train = pd.read_csv(os.path.join(DATA_PATH, "X_train.csv"))
    y_train = pd.read_csv(os.path.join(DATA_PATH, "y_train.csv")).values.ravel()
    print(f"   [OK] X_train shape: {X_train.shape}")
    print(f"   [OK] y_train shape: {y_train.shape}")
except Exception as e:
    print(f"   [ERROR] Error al cargar datos de entrenamiento: {e}")
    exit(1)

print("\n[2] Cargando datos de validación...")
try:
    X_validation = pd.read_csv(os.path.join(VAL_PATH, "X_validation.csv"))
    y_validation = pd.read_csv(os.path.join(VAL_PATH, "y_validation.csv")).values.ravel()
    print(f"   [OK] X_validation shape: {X_validation.shape}")
    print(f"   [OK] y_validation shape: {y_validation.shape}")
except Exception as e:
    print(f"   [ERROR] Error al cargar datos de validación: {e}")
    exit(1)

# ======================================================================
# 3. ENTRENAMIENTO DEL MODELO
# ======================================================================

print("\n[3] Entrenando Regresión Lineal con regularización Ridge...")

# Usar Ridge (L2) como regularización para evitar sobreajuste
# Alpha de 1.0 es un buen valor por defecto
model = Ridge(alpha=1.0, random_state=42)

try:
    model.fit(X_train, y_train)
    print("   [OK] Modelo entrenado exitosamente")
except Exception as e:
    print(f"   [ERROR] Error durante entrenamiento: {e}")
    exit(1)

# ======================================================================
# 4. EVALUACIÓN EN VALIDATION SET
# ======================================================================

print("\n[4] Evaluando modelo en VALIDATION SET (sin leakage)...")

try:
    # Predicciones en validation
    y_pred_val = model.predict(X_validation)
    
    # Cálculo de métricas
    mae = mean_absolute_error(y_validation, y_pred_val)
    mse = mean_squared_error(y_validation, y_pred_val)
    rmse = np.sqrt(mse)
    mape = mean_absolute_percentage_error(y_validation, y_pred_val)
    
    print(f"   [OK] MAE:  {mae:.4f} minutos")
    print(f"   [OK] MSE:  {mse:.4f}")
    print(f"   [OK] RMSE: {rmse:.4f} minutos")
    print(f"   [OK] MAPE: {mape:.4f} ({mape*100:.2f}%)")
    
except Exception as e:
    print(f"   [ERROR] Error durante evaluación: {e}")
    exit(1)

# ======================================================================
# 5. EXPORTACIÓN DEL MODELO
# ======================================================================

print("\n[5] Exportando modelo entrenado...")

try:
    joblib.dump(model, MODEL_PATH)
    print(f"   [OK] Modelo guardado en: {MODEL_PATH}")
except Exception as e:
    print(f"   [ERROR] Error al guardar modelo: {e}")
    exit(1)

# ======================================================================
# 6. EXPORTACIÓN DE MÉTRICAS CSV
# ======================================================================

print("\n[6] Exportando métricas al CSV consolidado...")

try:
    # Crear o actualizar CSV de métricas
    new_metric = pd.DataFrame({
        'model_name': ['Linear Regression'],
        'MAE': [mae],
        'MSE': [mse],
        'RMSE': [rmse],
        'MAPE': [mape],
        'n_train': [X_train.shape[0]],
        'n_validation': [X_validation.shape[0]]
    })
    
    # Si el archivo ya existe, leer y actualizar
    if os.path.exists(METRICS_CSV_PATH):
        existing_metrics = pd.read_csv(METRICS_CSV_PATH)
        # Remover fila de Linear Regression si existe (actualizar)
        existing_metrics = existing_metrics[existing_metrics['model_name'] != 'Linear Regression']
        # Concatenar con nuevas métricas
        metrics_df = pd.concat([existing_metrics, new_metric], ignore_index=True)
    else:
        metrics_df = new_metric
    
    # Guardar CSV
    metrics_df.to_csv(METRICS_CSV_PATH, index=False)
    print(f"   [OK] Métricas guardadas en: {METRICS_CSV_PATH}")
    
except Exception as e:
    print(f"   [ERROR] Error al guardar métricas: {e}")
    exit(1)

# ======================================================================
# 7. RESUMEN FINAL
# ======================================================================

print("\n" + "=" * 70)
print("RESUMEN: REGRESIÓN LINEAL")
print("=" * 70)
print(f"Modelo guardado:     {MODEL_PATH}")
print(f"Métricas guardadas:  {METRICS_CSV_PATH}")
print(f"Datos de validación: {X_validation.shape[0]} muestras")
print(f"Error RMSE:          {rmse:.4f} minutos")
print("=" * 70)
