# -*- coding: utf-8 -*-
# ======================================================================
# SCRIPT: ENTRENAMIENTO GLM (GENERALIZED LINEAR MODEL)
# Problema: Predicción de tiempo_real_min en logística última milla
# Familia: Gamma (apropiada para tiempos positivos)
# Link: log (estabiliza varianza)
# ======================================================================

import os
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Para GLM usamos statsmodels
import statsmodels.api as sm
from statsmodels.genmod.families import Gamma
from statsmodels.genmod.cov_struct import Independence
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import joblib
from pathlib import Path

# ======================================================================
# 1. CONFIGURACIÓN DE RUTAS
# ======================================================================

BASE_PATH = r"K:\Dashboard Delivery"
DATA_PATH = os.path.join(BASE_PATH, "data", "processed", "Train", "glm")
VAL_PATH = os.path.join(BASE_PATH, "data", "processed", "validation", "glm")
MODEL_PATH = os.path.join(BASE_PATH, "models", "glm_model.joblib")
METRICS_CSV_PATH = os.path.join(BASE_PATH, "trainig_models", "metrics", "model_metrics.csv")

# Crear directorio de métricas si no existe
os.makedirs(os.path.dirname(METRICS_CSV_PATH), exist_ok=True)

# ======================================================================
# 2. CARGA DE DATOS
# ======================================================================

print("=" * 70)
print("ENTRENAMIENTO: GLM (GENERALIZED LINEAR MODEL)")
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

print("\n[3] Entrenando GLM con familia Gamma e link log...")
print("   (Gamma es apropiada para tiempos positivos contínuos)")

try:
    # Añadir constante para el intercepto
    X_train_const = sm.add_constant(X_train)
    
    # Crear modelo GLM con familia Gamma e identity link (log también es opción)
    # Usando family.Gamma() por defecto con link log
    glm_model = sm.GLM(y_train, X_train_const, 
                       family=Gamma(sm.genmod.families.links.log()),
                       offset=None)
    
    # Entrenar
    result = glm_model.fit(maxiter=100, disp=0)
    
    print("   [OK] Modelo GLM entrenado exitosamente")
    
except Exception as e:
    print(f"   ✗ Error durante entrenamiento: {e}")
    # Fallback a GLM simple si hay error
    print("   [Fallback] Utilizando GLM simple...")
    try:
        X_train_const = sm.add_constant(X_train)
        glm_model = sm.GLM(y_train, X_train_const, family=Gamma())
        result = glm_model.fit(maxiter=100, disp=0)
        print("   [OK] Modelo GLM (simple) entrenado exitosamente")
    except Exception as e2:
        print(f"   [ERROR] Error en fallback: {e2}")
        exit(1)

# ======================================================================
# 4. EVALUACIÓN EN VALIDATION SET
# ======================================================================

print("\n[4] Evaluando modelo en VALIDATION SET (sin leakage)...")

try:
    # Predicciones en validation
    X_val_const = sm.add_constant(X_validation)
    y_pred_val = result.predict(X_val_const)
    
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
    # Guardar el resultado del GLM (GLMResults)
    joblib.dump(result, MODEL_PATH)
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
        'model_name': ['GLM (Gamma, log-link)'],
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
        # Remover fila de GLM si existe (actualizar)
        existing_metrics = existing_metrics[~existing_metrics['model_name'].str.contains('GLM', na=False)]
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
print("RESUMEN: GLM (GENERALIZED LINEAR MODEL)")
print("=" * 70)
print(f"Familia:             Gamma")
print(f"Link:                Log")
print(f"Modelo guardado:     {MODEL_PATH}")
print(f"Métricas guardadas:  {METRICS_CSV_PATH}")
print(f"Datos de validación: {X_validation.shape[0]} muestras")
print(f"Error RMSE:          {rmse:.4f} minutos")
print("=" * 70)
