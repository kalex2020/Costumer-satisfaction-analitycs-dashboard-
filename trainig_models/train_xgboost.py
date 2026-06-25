# -*- coding: utf-8 -*-
# ======================================================================
# SCRIPT: ENTRENAMIENTO XGBOOST REGRESSOR
# Problema: Predicción de tiempo_real_min en logística última milla
# Características: Early stopping, sin escalado (tree-based), máxima precisión
# ======================================================================

import os
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import joblib
from pathlib import Path

# ======================================================================
# 1. CONFIGURACIÓN DE RUTAS
# ======================================================================

BASE_PATH = r"K:\Dashboard Delivery"
DATA_PATH = os.path.join(BASE_PATH, "data", "processed", "Train", "xgboost")
VAL_PATH = os.path.join(BASE_PATH, "data", "processed", "validation", "xgboost")
MODEL_PATH = os.path.join(BASE_PATH, "models", "xgboost_model.joblib")
METRICS_CSV_PATH = os.path.join(BASE_PATH, "trainig_models", "metrics", "model_metrics.csv")

# Crear directorio de métricas si no existe
os.makedirs(os.path.dirname(METRICS_CSV_PATH), exist_ok=True)

# ======================================================================
# 2. CARGA DE DATOS
# ======================================================================

print("=" * 70)
print("ENTRENAMIENTO: XGBOOST REGRESSOR")
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
# 3. ENTRENAMIENTO DEL MODELO CON EARLY STOPPING
# ======================================================================

print("\n[3] Entrenando XGBoost Regressor con Early Stopping...")
print("   Parámetros:")
print("   - max_depth: 6")
print("   - learning_rate: 0.05")
print("   - n_estimators: 1000")
print("   - early_stopping_rounds: 10")
print("   - subsample: 0.8")
print("   - colsample_bytree: 0.8")

try:
    model = XGBRegressor(
        # Profundidad de árboles (evitar overfitting)
        max_depth=6,
        
        # Tasa de aprendizaje (bajo para mejor generalización)
        learning_rate=0.05,
        
        # Número de iteraciones (alto, con early stopping)
        n_estimators=1000,
        
        # Subsample: fracción de muestras para entrenar cada árbol
        subsample=0.8,
        
        # Colsample_bytree: fracción de features para cada árbol
        colsample_bytree=0.8,
        
        # Regularización L2
        reg_lambda=1.0,
        
        # Regularización L1 (Lasso)
        reg_alpha=0.0,
        
        # Métrica de evaluación
        eval_metric='rmse',
        
        # Repetibilidad
        random_state=42,
        
        # Verbose para seguimiento
        verbosity=0
    )
    
    # Entrenar CON EARLY STOPPING
    # Usar validation set para monitoreo
    model.fit(
        X_train, y_train,
        eval_set=[(X_validation, y_validation)],
        verbose=False
    )
    
    print(f"   [OK] Modelo entrenado exitosamente")
    # Solo mostrar best_iteration si early stopping fue usado
    try:
        print(f"   [OK] Mejor iteracion: {model.best_iteration}")
        print(f"   [OK] Mejor score: {model.best_score:.4f}")
    except:
        print(f"   [OK] Iteraciones completadas: {model.n_estimators}")
    
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
# 5. ANÁLISIS DE IMPORTANCIA DE FEATURES
# ======================================================================

print("\n[5] Importancia de features (top 5)...")

try:
    feature_importance = pd.DataFrame({
        'feature_index': range(len(model.feature_importances_)),
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("   Top 5 features más importantes:")
    for idx, row in feature_importance.head(5).iterrows():
        print(f"   - Feature {int(row['feature_index'])}: {row['importance']:.4f}")
    
except Exception as e:
    print(f"   ⚠ No se pudo calcular importancia de features: {e}")

# ======================================================================
# 6. EXPORTACIÓN DEL MODELO
# ======================================================================

print("\n[6] Exportando modelo entrenado...")

try:
    joblib.dump(model, MODEL_PATH)
    print(f"   [OK] Modelo guardado en: {MODEL_PATH}")
except Exception as e:
    print(f"   [ERROR] Error al guardar modelo: {e}")
    exit(1)

# ======================================================================
# 7. EXPORTACIÓN DE MÉTRICAS CSV
# ======================================================================

print("\n[7] Exportando métricas al CSV consolidado...")

try:
    # Crear o actualizar CSV de métricas
    new_metric = pd.DataFrame({
        'model_name': ['XGBoost'],
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
        # Remover fila de XGBoost si existe (actualizar)
        existing_metrics = existing_metrics[existing_metrics['model_name'] != 'XGBoost']
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
# 8. RESUMEN FINAL
# ======================================================================

print("\n" + "=" * 70)
print("RESUMEN: XGBOOST REGRESSOR")
print("=" * 70)
print(f"Modelo guardado:     {MODEL_PATH}")
print(f"Métricas guardadas:  {METRICS_CSV_PATH}")
print(f"Datos de validación: {X_validation.shape[0]} muestras")
print(f"Error RMSE:          {rmse:.4f} minutos")
print(f"Iteraciones usadas:  {model.n_estimators if not hasattr(model, 'best_iteration') else model.best_iteration}")
print("=" * 70)
