# -*- coding: utf-8 -*-
"""
Servicio de métricas agregadas para KPIs y flashcards.
Lee archivo de métricas y calcula agregados históricos.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime

import sys
sys.path.insert(0, str(__file__).replace("\\", "/").rsplit("/", 2)[0])

from config import (
    METRICS_CSV_PATH,
    DEFAULT_MODEL,
    VALIDATION_DATA_PATHS,
    DATASET_MAESTRO_PATH,
    TARGET_VARIABLE,
)

# ======================================================================
# CONFIGURACIÓN DE LOGGING
# ======================================================================

logger = logging.getLogger(__name__)

# ======================================================================
# SERVICIO DE MÉTRICAS
# ======================================================================

class MetricsService:
    """
    Servicio para cálculo de métricas agregadas.
    Lee archivos de validación y calcula indicadores para dashboards.
    """
    
    def __init__(self):
        """Inicializa el servicio."""
        self.model_metrics: Dict[str, Dict[str, float]] = {}
        self.validation_data: Dict[str, pd.DataFrame] = {}
        self.maestro_data: Optional[pd.DataFrame] = None
        
        # Cargar datos
        self._load_model_metrics()
        self._load_validation_data()
        self._load_maestro_data()
    
    def _load_model_metrics(self) -> None:
        """Carga las métricas de modelos desde CSV consolidado."""
        try:
            logger.info(f"Cargando métricas desde: {METRICS_CSV_PATH}")
            metrics_df = pd.read_csv(METRICS_CSV_PATH)
            
            # Convertir a diccionario por modelo
            for _, row in metrics_df.iterrows():
                model_name = row['model_name'].lower()
                # Simplificar nombres
                if 'linear' in model_name:
                    model_name = 'linear'
                elif 'glm' in model_name.lower():
                    model_name = 'glm'
                elif 'histgradient' in model_name.lower():
                    model_name = 'histgb'
                elif 'xgboost' in model_name.lower():
                    model_name = 'xgboost'
                
                self.model_metrics[model_name] = {
                    'mae': float(row['MAE']),
                    'mse': float(row['MSE']),
                    'rmse': float(row['RMSE']),
                    'mape': float(row['MAPE']),
                    'n_train': int(row['n_train']),
                    'n_validation': int(row['n_validation']),
                }
            
            logger.info(f"[OK] Métricas cargadas para {len(self.model_metrics)} modelos")
        
        except FileNotFoundError:
            logger.error(f"Archivo de métricas no encontrado: {METRICS_CSV_PATH}")
        except Exception as e:
            logger.error(f"Error al cargar métricas: {str(e)}")
    
    def _load_validation_data(self) -> None:
        """Carga datos de validación por modelo."""
        for model_name, data_path in VALIDATION_DATA_PATHS.items():
            try:
                logger.info(f"Cargando datos de validación para {model_name}...")
                
                X_val = pd.read_csv(f"{data_path}/X_validation.csv")
                y_val = pd.read_csv(f"{data_path}/y_validation.csv").values.ravel()
                
                # Combinar en un DataFrame
                data = X_val.copy()
                data[TARGET_VARIABLE] = y_val
                
                self.validation_data[model_name] = data
                logger.info(f"[OK] {len(data)} registros de validación para {model_name}")
            
            except Exception as e:
                logger.warning(f"No se pudo cargar validación para {model_name}: {e}")
    
    def _load_maestro_data(self) -> None:
        """Carga dataset maestro para análisis históricos."""
        try:
            logger.info(f"Cargando dataset maestro...")
            self.maestro_data = pd.read_csv(DATASET_MAESTRO_PATH)
            logger.info(f"[OK] {len(self.maestro_data)} registros en dataset maestro")
        except Exception as e:
            logger.warning(f"No se pudo cargar dataset maestro: {e}")
    
    def get_model_metrics(self, model_name: Optional[str] = None) -> Dict[str, float]:
        """
        Retorna métricas (MAE, MSE, RMSE, MAPE) de un modelo.
        
        Args:
            model_name: Nombre del modelo
        
        Returns:
            Diccionario con métricas
        """
        if model_name is None:
            model_name = DEFAULT_MODEL
        
        model_name_lower = model_name.lower()
        
        if model_name_lower not in self.model_metrics:
            logger.warning(f"Métricas no disponibles para: {model_name}")
            return {}
        
        return self.model_metrics[model_name_lower]
    
    def calculate_flashcards(self, model_name: Optional[str] = None) -> Dict[str, any]:
        """
        Calcula métricas para flashcards (KPIs).
        
        Args:
            model_name: Modelo a analizar
        
        Returns:
            Diccionario con flashcards
        """
        if model_name is None:
            model_name = DEFAULT_MODEL
        
        model_name_lower = model_name.lower()
        
        flashcards = {
            "timestamp": datetime.now(),
            "model_name": model_name_lower,
            "flashcards": [],
            "n_records": 0,
        }
        
        # Obtener datos de validación
        if model_name_lower in self.validation_data:
            data = self.validation_data[model_name_lower]
            n_records = len(data)
            
            flashcards["n_records"] = n_records
            
            # Extraer variable target
            if TARGET_VARIABLE in data.columns:
                y_real = data[TARGET_VARIABLE].values
                
                # Flashcard 1: ETA Promedio
                eta_mean = float(np.mean(y_real))
                flashcards["flashcards"].append({
                    "label": "ETA Promedio",
                    "value": round(eta_mean, 2),
                    "unit": "min",
                    "trend": None
                })
                
                # Flashcard 2: ETA Máximo
                eta_max = float(np.max(y_real))
                flashcards["flashcards"].append({
                    "label": "ETA Máximo",
                    "value": round(eta_max, 2),
                    "unit": "min",
                    "trend": None
                })
                
                # Flashcard 3: ETA Mínimo
                eta_min = float(np.min(y_real))
                flashcards["flashcards"].append({
                    "label": "ETA Mínimo",
                    "value": round(eta_min, 2),
                    "unit": "min",
                    "trend": None
                })
                
                # Flashcard 4: Desviación Estándar
                eta_std = float(np.std(y_real))
                flashcards["flashcards"].append({
                    "label": "Desviación Estándar",
                    "value": round(eta_std, 2),
                    "unit": "min",
                    "trend": None
                })
        
        # Agregar métricas del modelo
        model_metrics = self.get_model_metrics(model_name)
        if model_metrics:
            # Flashcard 5: MAE (Error Promedio)
            mae = model_metrics.get('mae', 0.0)
            flashcards["flashcards"].append({
                "label": "Error Promedio (MAE)",
                "value": round(float(mae), 2),
                "unit": "min",
                "trend": "down"  # Bajo es bueno
            })
            
            # Flashcard 6: RMSE
            rmse = model_metrics.get('rmse', 0.0)
            flashcards["flashcards"].append({
                "label": "RMSE",
                "value": round(float(rmse), 2),
                "unit": "min",
                "trend": "down"
            })
            
            # Flashcard 7: MAPE (%)
            mape = model_metrics.get('mape', 0.0) * 100
            flashcards["flashcards"].append({
                "label": "Error Porcentual (MAPE)",
                "value": round(float(mape), 2),
                "unit": "%",
                "trend": "down"
            })
            
            # Flashcard 8: N Validación
            n_val = model_metrics.get('n_validation', 0)
            flashcards["flashcards"].append({
                "label": "Registros Validación",
                "value": int(n_val),
                "unit": "registros",
                "trend": None
            })
        
        logger.info(f"Flashcards calculadas para {model_name}: {len(flashcards['flashcards'])} métricas")
        return flashcards
    
    def get_all_models_metrics(self) -> List[Dict[str, any]]:
        """Retorna métricas de todos los modelos."""
        all_metrics = []
        
        for model_name in self.model_metrics.keys():
            metrics = self.get_model_metrics(model_name)
            if metrics:
                all_metrics.append({
                    "model_name": model_name,
                    **metrics
                })
        
        return all_metrics
    
    def get_best_model(self) -> Tuple[str, float]:
        """
        Retorna el modelo con mejor desempeño (menor RMSE).
        
        Returns:
            Tupla (model_name, rmse)
        """
        best_model = None
        best_rmse = float('inf')
        
        for model_name, metrics in self.model_metrics.items():
            rmse = metrics.get('rmse', float('inf'))
            if rmse < best_rmse:
                best_rmse = rmse
                best_model = model_name
        
        if best_model is None:
            best_model = DEFAULT_MODEL
            best_rmse = self.model_metrics.get(best_model, {}).get('rmse', 0.0)
        
        logger.info(f"Mejor modelo: {best_model} (RMSE: {best_rmse:.4f})")
        return best_model, float(best_rmse)


# ======================================================================
# INSTANCIA GLOBAL DEL SERVICIO (Singleton Pattern)
# ======================================================================

_metrics_service: Optional[MetricsService] = None

def get_metrics_service() -> MetricsService:
    """
    Retorna la instancia singleton del servicio de métricas.
    Crea la instancia en la primera llamada.
    """
    global _metrics_service
    
    if _metrics_service is None:
        logger.info("Inicializando MetricsService...")
        _metrics_service = MetricsService()
    
    return _metrics_service


if __name__ == "__main__":
    # Prueba local
    print("[TEST] Inicializando servicio de métricas...")
    service = get_metrics_service()
    
    print("\n[TEST] Métricas de todos los modelos:")
    metrics = service.get_all_models_metrics()
    for m in metrics:
        print(f"  {m['model_name']}: RMSE={m['rmse']:.4f}, MAE={m['mae']:.4f}")
    
    print("\n[TEST] Mejor modelo:")
    best_model, best_rmse = service.get_best_model()
    print(f"  Modelo: {best_model}, RMSE: {best_rmse:.4f}")
    
    print("\n[TEST] Flashcards para modelo linear:")
    flashcards = service.calculate_flashcards("linear")
    print(f"  N Registros: {flashcards['n_records']}")
    print(f"  Flashcards calculadas: {len(flashcards['flashcards'])}")
    for fc in flashcards['flashcards']:
        print(f"    - {fc['label']}: {fc['value']} {fc['unit']}")
