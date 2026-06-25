# -*- coding: utf-8 -*-
"""
Configuración centralizada del backend.
Define rutas, parámetros y constantes.
"""

import os
from pathlib import Path

# ======================================================================
# RUTAS BASE
# ======================================================================

# Obtener ruta base del proyecto
BASE_DIR = Path(__file__).resolve().parent.parent
BACKEND_DIR = BASE_DIR / "backend"
MODELS_DIR = BASE_DIR / "models"
METRICS_DIR = BASE_DIR / "trainig_models" / "metrics"
DATA_DIR = BACKEND_DIR / "data"

# ======================================================================
# RUTAS DE MODELOS
# ======================================================================

MODEL_PATHS = {
    "linear": str(MODELS_DIR / "linear_model.joblib"),
    "glm": str(MODELS_DIR / "glm_model.joblib"),
    "histgb": str(MODELS_DIR / "histgb_model.joblib"),
    "xgboost": str(MODELS_DIR / "xgboost_model.joblib"),
}

# Modelo por defecto (el de mejor desempeño en validación)
DEFAULT_MODEL = "linear"

# ======================================================================
# RUTAS DE DATOS
# ======================================================================

METRICS_CSV_PATH = str(METRICS_DIR / "model_metrics.csv")

# Dataset maestro para análisis históricos
DATASET_MAESTRO_PATH = str(BASE_DIR / "data" / "processed" / "dataset_maestro_intrak_bogota.csv")

# Datos de validación para cálculo de métricas
VALIDATION_DATA_PATHS = {
    "linear": str(BASE_DIR / "data" / "processed" / "validation" / "linear"),
    "glm": str(BASE_DIR / "data" / "processed" / "validation" / "glm"),
    "histgb": str(BASE_DIR / "data" / "processed" / "validation" / "histgb"),
    "xgboost": str(BASE_DIR / "data" / "processed" / "validation" / "xgboost"),
}

# ======================================================================
# CONFIGURACIÓN DE LA API
# ======================================================================

API_TITLE = "Dashboard Backend - Logística Última Milla"
API_VERSION = "0.1.0"
API_DESCRIPTION = "Backend para predicción de ETA y cálculo de KPIs en logística de última milla"

# ======================================================================
# CONFIGURACIÓN DE PREDICCIÓN
# ======================================================================

# Nombre del target que predecimos
TARGET_VARIABLE = "tiempo_real_min"

# Descripción de la variable target
TARGET_DESCRIPTION = "Tiempo real de entrega en minutos"

# ======================================================================
# CONFIGURACIÓN DE FILTROS Y VALIDACIÓN
# ======================================================================

# Valores posibles para localidades (extraer del dataset en runtime)
VALID_LOCALIDADES = [
    "Usaquén", "Chapinero", "Santa Fe", "San Cristóbal",
    "Usme", "Kennedy", "Fontibón", "Engativá", "Suba",
    "Barrios Unidos", "Teusaquillo", "Los Mártires", "Antonio Nariño",
    "Puente Aranda", "La Candelaria", "Rafael Uribe Umaña", "Ciudad Bolívar",
    "Sumapaz"
]

# Rangos de validación para features
FEATURE_RANGES = {
    "distancia_km": (0.1, 50.0),
    "nivel_trafico": (1, 5),
    "lluvia_mm": (0.0, 100.0),
    "estrato": (1, 6),
    "pendiente_promedio": (0.0, 30.0),
}

# ======================================================================
# VALIDACIÓN DE DIRECTORIOS
# ======================================================================

def validate_paths():
    """Valida que existan todas las rutas necesarias."""
    errors = []
    
    # Validar directorios
    for name, path in {
        "MODELS_DIR": MODELS_DIR,
        "METRICS_DIR": METRICS_DIR,
        "BASE_DIR": BASE_DIR,
    }.items():
        if not path.exists():
            errors.append(f"Directorio faltante: {name} ({path})")
    
    # Validar rutas de modelos
    for model_name, model_path in MODEL_PATHS.items():
        if not os.path.exists(model_path):
            errors.append(f"Modelo faltante: {model_name} ({model_path})")
    
    # Validar archivo de métricas
    if not os.path.exists(METRICS_CSV_PATH):
        errors.append(f"Archivo de métricas faltante ({METRICS_CSV_PATH})")
    
    if errors:
        for error in errors:
            print(f"[WARNING] {error}")
    
    return len(errors) == 0

if __name__ == "__main__":
    print("[INFO] Validando configuración de rutas...")
    if validate_paths():
        print("[OK] Todas las rutas validadas correctamente")
    else:
        print("[ERROR] Algunas rutas no existen")
