# -*- coding: utf-8 -*-
"""
Servicio de predicción de ETA con preprocesamiento automático.
Carga modelos y preprocessors, realiza inferencias.
"""

import joblib
import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Optional, Tuple
from datetime import datetime
from pathlib import Path

import sys
sys.path.insert(0, str(__file__).replace("\\", "/").rsplit("/", 2)[0])

from config import MODEL_PATHS, DEFAULT_MODEL, TARGET_VARIABLE

# ======================================================================
# CONFIGURACIÓN DE LOGGING
# ======================================================================

logger = logging.getLogger(__name__)

# ======================================================================
# DEFINICIÓN DE FEATURES: DEBE COINCIDIR CON prepare_dataset.py
# ======================================================================

NUMERIC_FEATURES = [
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

CATEGORICAL_FEATURES = [
    "localidad",
    "tipo_via",
    "tipo_zona",
    "vehiculo"
]

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# ======================================================================
# SERVICIO DE PREDICCIÓN
# ======================================================================

class PredictionService:
    """
    Servicio para cargar modelos y realizar predicciones.
    - Carga modelos desde .joblib
    - Carga preprocessors fitting en datos de entrenamiento
    - Aplica preprocesamiento idéntico al entrenamiento
    """
    
    def __init__(self):
        """Inicializa el servicio y carga modelos + preprocessors."""
        self.models: Dict[str, Any] = {}
        self.model_metadata: Dict[str, Dict[str, Any]] = {}
        self.preprocessor_linear = None
        self.preprocessor_boost = None
        
        logger.info("Inicializando PredictionService...")
        self.load_models()
        self.load_preprocessors()
        logger.info("PredictionService inicializado correctamente")
    
    def load_models(self) -> Dict[str, bool]:
        """Carga todos los modelos desde .joblib."""
        models_status = {}
        
        for model_name, model_path in MODEL_PATHS.items():
            try:
                logger.info(f"Cargando modelo: {model_name} desde {model_path}")
                model = joblib.load(model_path)
                self.models[model_name] = model
                models_status[model_name] = True
                logger.info(f"[OK] Modelo cargado: {model_name}")
            except FileNotFoundError:
                error_msg = f"Archivo no encontrado: {model_path}"
                logger.error(error_msg)
                models_status[model_name] = False
            except Exception as e:
                error_msg = f"Error al cargar {model_name}: {str(e)}"
                logger.error(error_msg)
                models_status[model_name] = False
        
        logger.info(f"Modelos cargados: {sum(models_status.values())}/{len(MODEL_PATHS)}")
        return models_status
    
    def load_preprocessors(self):
        """Carga los preprocessors desde .joblib generados en entrenamiento."""
        try:
            base_dir = Path(__file__).parent.parent.parent
            preprocessor_dir = base_dir / "models" / "preprocessors"
            
            # Cargar preprocessor Linear/GLM
            linear_path = preprocessor_dir / "preprocessor_linear_glm.joblib"
            if linear_path.exists():
                self.preprocessor_linear = joblib.load(str(linear_path))
                logger.info("[OK] Preprocessor Linear/GLM cargado")
            else:
                logger.warning(f"Preprocessor Linear/GLM no encontrado: {linear_path}")
            
            # Cargar preprocessor HistGB/XGBoost
            boost_path = preprocessor_dir / "preprocessor_histgb_xgboost.joblib"
            if boost_path.exists():
                self.preprocessor_boost = joblib.load(str(boost_path))
                logger.info("[OK] Preprocessor HistGB/XGBoost cargado")
            else:
                logger.warning(f"Preprocessor HistGB/XGBoost no encontrado: {boost_path}")
                
        except Exception as e:
            logger.error(f"Error cargando preprocessors: {str(e)}")
    
    def get_available_models(self) -> Dict[str, bool]:
        """Retorna disponibilidad de cada modelo."""
        return {
            model_name: model_name in self.models
            for model_name in MODEL_PATHS.keys()
        }
    
    def validate_model(self, model_name: str) -> bool:
        """Valida que el modelo exista y esté cargado."""
        if model_name not in self.models:
            logger.error(f"Modelo no disponible: {model_name}")
            return False
        return True
    
    def prepare_input_data(self, input_data: Dict[str, Any]) -> pd.DataFrame:
        """
        Prepara y valida los datos de entrada.
        - Agrega características faltantes con valores por defecto
        - Calcula eficiencia_estimacion si no existe
        - Retorna DataFrame con todas las features requeridas en el orden correcto
        """
        # Valores por defecto
        defaults = {
            "estrato": 3,
            "distancia_km": 5.0,
            "nivel_trafico": 2,
            "lluvia_mm": 0.0,
            "pendiente_promedio": 0.5,
            "zona_riesgo": 0,
            "cliente_ausente": 0,
            "intentos_entrega": 1,
            "tiempo_estimado_min": 15.0,
            "anio": 2024,
            "mes": 1,
            "dia": 1,
            "dia_semana": 0,
            "localidad": "Kennedy",
            "tipo_via": "Calle",
            "tipo_zona": "Urbana",
            "vehiculo": "Moto"
        }
        
        # Combinar input_data con defaults
        data = defaults.copy()
        data.update(input_data)
        
        # Calcular eficiencia_estimacion si no existe
        if "eficiencia_estimacion" not in data or pd.isna(data.get("eficiencia_estimacion")):
            if data.get("distancia_km", 0) > 0:
                data["eficiencia_estimacion"] = data["tiempo_estimado_min"] / data["distancia_km"]
            else:
                data["eficiencia_estimacion"] = 0.0
        
        # Reemplazar infinitos
        if np.isinf(data.get("eficiencia_estimacion", 0)):
            data["eficiencia_estimacion"] = 0.0
        
        # Crear DataFrame con todas las columnas en el orden definido
        df = pd.DataFrame({col: [data.get(col, defaults.get(col))] for col in ALL_FEATURES})
        
        logger.debug(f"Datos preparados: {df.shape}")
        return df
    
    def preprocess_linear(self, df: pd.DataFrame) -> np.ndarray:
        """Preprocesa datos para Linear/GLM usando el preprocessor cargado."""
        if self.preprocessor_linear is None:
            logger.error("Preprocessor Linear/GLM no disponible")
            raise RuntimeError("Preprocessor Linear/GLM no disponible")
        
        try:
            # Aplicar transformación (StandardScaler + OneHotEncoder)
            X_transformed = self.preprocessor_linear.transform(df)
            logger.debug(f"Datos preprocesados Linear: {X_transformed.shape}")
            return X_transformed
        except Exception as e:
            logger.error(f"Error en preprocesamiento linear: {str(e)}")
            raise
    
    def preprocess_histgb_xgboost(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocesa datos para HistGB/XGBoost (ordinal encoding)."""
        df = df.copy()
        
        if self.preprocessor_boost is None:
            logger.warning("Preprocessor HistGB/XGBoost no disponible, usando datos crudos")
            return df
        
        # Extraer información del preprocessor (sample data con mappings)
        sample_data = self.preprocessor_boost.get("training_data_sample")
        
        if sample_data is None or sample_data.empty:
            logger.warning("Datos de muestra no disponibles en preprocessor")
            return df
        
        try:
            # Crear mappings desde datos de muestra
            for col in CATEGORICAL_FEATURES:
                if col in df.columns and col in sample_data.columns:
                    # Obtener valores únicos ordenados (mismo que durante fit)
                    unique_vals = sorted(sample_data[col].unique())
                    mapping = {v: i for i, v in enumerate(unique_vals)}
                    
                    # Aplicar mapping (-1 para valores desconocidos)
                    df[col] = df[col].map(mapping).fillna(-1).astype(int)
            
            logger.debug(f"Datos preprocesados HistGB/XGBoost: {df.shape}")
            return df
        except Exception as e:
            logger.error(f"Error en preprocesamiento HistGB/XGBoost: {str(e)}")
            raise
    
    def predict(
        self,
        input_data: Dict[str, Any],
        model_name: Optional[str] = None
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Realiza una predicción de ETA.
        
        Args:
            input_data: Diccionario con features
            model_name: Nombre del modelo a usar (default: LINEAR)
        
        Returns:
            Tupla (predicción, metadata)
        """
        # Usar modelo default si no se especifica
        if model_name is None:
            model_name = DEFAULT_MODEL
        
        # Validar modelo
        if not self.validate_model(model_name):
            raise ValueError(f"Modelo no disponible: {model_name}")
        
        try:
            # Preparar features en un dataframe limpio
            df_raw = self.prepare_input_data(input_data)
            
            # Preprocesar según tipo de modelo
            if model_name in ["linear", "glm"]:
                # Usar preprocessor Linear/GLM (StandardScaler + OneHotEncoder)
                X = self.preprocess_linear(df_raw)
            elif model_name in ["histgb", "xgboost"]:
                # Usar preprocesamiento HistGB/XGBoost (ordinal encoding)
                df_processed = self.preprocess_histgb_xgboost(df_raw)
                X = df_processed.values
            else:
                raise ValueError(f"Tipo de modelo desconocido: {model_name}")
            
            # Obtener modelo y realizar predicción
            model = self.models[model_name]
            prediction = model.predict(X)[0]
            
            # Asegurar que la predicción es positiva
            prediction = max(float(prediction), 0.1)
            
            # Preparar metadata
            metadata = {
                "model_name": model_name,
                "timestamp": datetime.now().isoformat(),
                "prediction_value": float(prediction),
                "input_features_raw": input_data,
            }
            
            logger.debug(f"Predicción exitosa con {model_name}: {prediction:.2f} min")
            return float(prediction), metadata
            
        except Exception as e:
            logger.error(f"Error durante predicción: {str(e)}")
            raise RuntimeError(f"Error en predicción: {str(e)}")
    
    def batch_predict(
        self,
        input_data_list: list,
        model_name: Optional[str] = None
    ) -> list:
        """
        Realiza predicciones en batch (múltiples registros).
        
        Args:
            input_data_list: Lista de diccionarios con features
            model_name: Modelo a usar
        
        Returns:
            Lista de predicciones
        """
        if model_name is None:
            model_name = DEFAULT_MODEL
        
        if not self.validate_model(model_name):
            raise ValueError(f"Modelo no disponible: {model_name}")
        
        try:
            predictions = []
            for item in input_data_list:
                pred, _ = self.predict(item, model_name)
                predictions.append(pred)
            
            logger.debug(f"Batch predictions: {len(predictions)} registros")
            return predictions
            
        except Exception as e:
            logger.error(f"Error en batch predict: {str(e)}")
            raise RuntimeError(f"Error en batch predict: {str(e)}")


# ======================================================================
# INSTANCIA GLOBAL DEL SERVICIO (Singleton Pattern)
# ======================================================================

_prediction_service: Optional[PredictionService] = None

def get_prediction_service() -> PredictionService:
    """
    Retorna la instancia singleton del servicio de predicción.
    Crea la instancia en la primera llamada.
    """
    global _prediction_service
    
    if _prediction_service is None:
        logger.info("Inicializando PredictionService...")
        _prediction_service = PredictionService()
    
    return _prediction_service


if __name__ == "__main__":
    # Prueba local
    print("[TEST] Inicializando servicio...")
    service = get_prediction_service()
    
    print("[TEST] Modelos disponibles:")
    print(service.get_available_models())
    
    # Prueba de predicción
    test_input = {
        "localidad": "Kennedy",
        "distancia_km": 5.5,
        "nivel_trafico": 3,
        "lluvia_mm": 2.0,
        "estrato": 3,
        "tipo_via": "Calle",
        "tipo_zona": "Urbana",
        "vehiculo": "Moto",
        "tiempo_estimado_min": 15.0,
    }
    
    print("\n[TEST] Realizando predicciones...")
    for model in ["linear", "glm", "histgb", "xgboost"]:
        try:
            pred, meta = service.predict(test_input, model)
            print(f"[OK] {model}: {pred:.2f} minutos")
        except Exception as e:
            print(f"[ERROR] {model}: {e}")
