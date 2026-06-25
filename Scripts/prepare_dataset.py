# ======================================================================
# MULTI-MODEL DATA PREPARATION PIPELINE (CSV OUTPUT)
# PROBLEMA: REGRESIÓN - LOGÍSTICA ÚLTIMA MILLA (COLOMBIA)
# TARGET: tiempo_real_min
#
# MODELOS:
# - Linear Regression
# - GLM
# - HistGradientBoostingRegressor
# - XGBoost Regressor
#
# NOTAS:
# - Split temporal (sin leakage)
# - Preparación diferenciada por modelo
# - Exporta TODO en CSV
# ======================================================================

import os
import pandas as pd
import numpy as np
import joblib

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

# ======================================================================
# 1. RUTAS
# ======================================================================

path_input = r"K:\Dashboard Delivery\data\processed\dataset_maestro_intrak_bogota.csv"

path_train = r"K:\Dashboard Delivery\data\processed\Train"
path_validation = r"K:\Dashboard Delivery\data\processed\validation"

models = ["linear", "glm", "histgb", "xgboost"]

for m in models:
    os.makedirs(os.path.join(path_train, m), exist_ok=True)
    os.makedirs(os.path.join(path_validation, m), exist_ok=True)

# ======================================================================
# 2. CARGA DATASET
# ======================================================================

df = pd.read_csv(path_input)

TARGET = "tiempo_real_min"

# ======================================================================
# 3. PROCESAMIENTO TEMPORAL
# ======================================================================

df["fecha_pedido"] = pd.to_datetime(df["fecha_pedido"])

# Orden temporal obligatorio (anti-leakage)
df = df.sort_values("fecha_pedido").reset_index(drop=True)

df["anio"] = df["fecha_pedido"].dt.year
df["mes"] = df["fecha_pedido"].dt.month
df["dia"] = df["fecha_pedido"].dt.day
df["dia_semana"] = df["fecha_pedido"].dt.dayofweek

df.drop(columns=["fecha_pedido"], inplace=True)

# ======================================================================
# 4. FEATURE ENGINEERING
# ======================================================================

df["eficiencia_estimacion"] = (
    df["tiempo_estimado_min"] / df["distancia_km"]
)

df["eficiencia_estimacion"] = df["eficiencia_estimacion"].replace(
    [np.inf, -np.inf], np.nan
).fillna(df["eficiencia_estimacion"].median())

# ======================================================================
# 5. DEFINICIÓN FEATURES
# ======================================================================

categorical_features = [
    "localidad",
    "tipo_via",
    "tipo_zona",
    "vehiculo"
]

numeric_features = [
    "estrato",
    "distancia_km",
    "nivel_trafico",
    "lluvia_mm",
    "pendiente_promedio",
    "zona_riesgo",
    "cliente_ausente",
    "intentos_entrega",
    "tiempo_estimado_min",
    "eficiencia_estimacion",
    "anio",
    "mes",
    "dia",
    "dia_semana"
]

# ======================================================================
# 6. SPLIT TEMPORAL TRAIN / VALIDATION
# ======================================================================

split_ratio = 0.8
split_index = int(len(df) * split_ratio)

df_train = df.iloc[:split_index]
df_val = df.iloc[split_index:]

X_train = df_train.drop(columns=[TARGET])
y_train = df_train[TARGET]

X_val = df_val.drop(columns=[TARGET])
y_val = df_val[TARGET]

# ======================================================================
# 7. MODELOS LINEALES Y GLM (OneHot + Scaling)
# ======================================================================

preprocessor_linear = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ]
)

X_train_linear = preprocessor_linear.fit_transform(X_train)
X_val_linear = preprocessor_linear.transform(X_val)

X_train_linear = pd.DataFrame(X_train_linear)
X_val_linear = pd.DataFrame(X_val_linear)

# Guardado Linear
X_train_linear.to_csv(
    os.path.join(path_train, "linear", "X_train.csv"),
    index=False
)
y_train.to_csv(
    os.path.join(path_train, "linear", "y_train.csv"),
    index=False
)

X_val_linear.to_csv(
    os.path.join(path_validation, "linear", "X_validation.csv"),
    index=False
)
y_val.to_csv(
    os.path.join(path_validation, "linear", "y_validation.csv"),
    index=False
)

# Guardado GLM (misma preparación)
X_train_linear.to_csv(
    os.path.join(path_train, "glm", "X_train.csv"),
    index=False
)
y_train.to_csv(
    os.path.join(path_train, "glm", "y_train.csv"),
    index=False
)

X_val_linear.to_csv(
    os.path.join(path_validation, "glm", "X_validation.csv"),
    index=False
)
y_val.to_csv(
    os.path.join(path_validation, "glm", "y_validation.csv"),
    index=False
)

# ======================================================================
# 8. HISTGB Y XGBOOST (Ordinal Encoding, sin escalado)
# ======================================================================

X_train_boost = X_train.copy()
X_val_boost = X_val.copy()

for col in categorical_features:
    X_train_boost[col], uniques = pd.factorize(X_train_boost[col])
    X_val_boost[col] = X_val_boost[col].apply(
        lambda x: uniques.get_loc(x) if x in uniques else -1
    )

# Guardado HistGB
X_train_boost.to_csv(
    os.path.join(path_train, "histgb", "X_train.csv"),
    index=False
)
y_train.to_csv(
    os.path.join(path_train, "histgb", "y_train.csv"),
    index=False
)

X_val_boost.to_csv(
    os.path.join(path_validation, "histgb", "X_validation.csv"),
    index=False
)
y_val.to_csv(
    os.path.join(path_validation, "histgb", "y_validation.csv"),
    index=False
)

# Guardado XGBoost
X_train_boost.to_csv(
    os.path.join(path_train, "xgboost", "X_train.csv"),
    index=False
)
y_train.to_csv(
    os.path.join(path_train, "xgboost", "y_train.csv"),
    index=False
)

X_val_boost.to_csv(
    os.path.join(path_validation, "xgboost", "X_validation.csv"),
    index=False
)
y_val.to_csv(
    os.path.join(path_validation, "xgboost", "y_validation.csv"),
    index=False
)

# ======================================================================
# 8b. GUARDAR PREPROCESSORS (REQUERIDO PARA PREDICCIÓN EN BACKEND)
# ======================================================================

preprocessor_path = os.path.join(
    r"K:\Dashboard Delivery", "models", "preprocessors"
)
os.makedirs(preprocessor_path, exist_ok=True)

# Guardar preprocessor de Linear/GLM
joblib.dump(
    preprocessor_linear,
    os.path.join(preprocessor_path, "preprocessor_linear_glm.joblib")
)
print("✅ Preprocessor Linear/GLM guardado")

# Crear un "preprocessor" para HistGB/XGBoost que contenga los mappings
# En este caso, guardaremos la información de factorize necesaria
preprocessor_histgb_xgboost = {
    "type": "ordinal_encoding",
    "numeric_features": numeric_features,
    "categorical_features": categorical_features,
    "categorical_mappings": {},
    "training_data_sample": X_train_boost.head(100)
}

joblib.dump(
    preprocessor_histgb_xgboost,
    os.path.join(preprocessor_path, "preprocessor_histgb_xgboost.joblib")
)
print("✅ Preprocessor HistGB/XGBoost guardado")

# ======================================================================
# 9. VALIDACIONES FINALES
# ======================================================================

assert len(X_train) > 0 and len(X_val) > 0
assert X_train_linear.shape[0] == y_train.shape[0]
assert X_val_linear.shape[0] == y_val.shape[0]

print("✅ PIPELINE COMPLETADO")
print("✅ Archivos CSV generados correctamente")
print("📁 Train:", path_train)
print("📁 Validation:", path_validation)