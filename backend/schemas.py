# -*- coding: utf-8 -*-
"""
Esquemas Pydantic para validación de request/response.
Define los contratos de la API REST.
"""

from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from datetime import datetime

# ======================================================================
# SCHEMAS DE PREDICCIÓN
# ======================================================================

class PredictionInput(BaseModel):
    """
    Schema para solicitudes de predicción de ETA.
    Contiene las variables operativas necesarias.
    """
    
    localidad: str = Field(
        ...,
        description="Localidad de entrega (ej: Kennedy)"
    )
    
    distancia_km: float = Field(
        ...,
        description="Distancia en kilómetros",
        gt=0.1, lt=50.0
    )
    
    nivel_trafico: int = Field(
        ...,
        description="Nivel de tráfico (1-5)",
        ge=1, le=5
    )
    
    lluvia_mm: float = Field(
        default=0.0,
        description="Precipitación en milímetros",
        ge=0.0, le=100.0
    )
    
    estrato: int = Field(
        default=3,
        description="Estrato socioeconómico (1-6)",
        ge=1, le=6
    )
    
    tipo_via: str = Field(
        default="Calle",
        description="Tipo de vía (Calle, Carrera, Diagonal, etc.)"
    )
    
    tipo_zona: str = Field(
        default="Urbana",
        description="Tipo de zona (Urbana, Suburbana, etc.)"
    )
    
    vehiculo: str = Field(
        default="Moto",
        description="Tipo de vehículo (Moto, Bicicleta, Auto)"
    )
    
    tiempo_estimado_min: float = Field(
        ...,
        description="Tiempo estimado por el algoritmo de ruteo (minutos)",
        gt=0, lt=500
    )
    
    modelo: str = Field(
        default="linear",
        description="Modelo a usar para predicción (linear, glm, histgb, xgboost)"
    )
    
    class Config:
        example = {
            "localidad": "Kennedy",
            "distancia_km": 5.5,
            "nivel_trafico": 3,
            "lluvia_mm": 2.0,
            "estrato": 3,
            "tipo_via": "Calle",
            "tipo_zona": "Urbana",
            "vehiculo": "Moto",
            "tiempo_estimado_min": 15.0,
            "modelo": "linear"
        }


class PredictionOutput(BaseModel):
    """
    Schema para respuestas de predicción.
    Contiene la predicción y metadata.
    """
    
    timestamp: datetime
    model_name: str
    prediction: float = Field(
        description="Predicción de tiempo real en minutos"
    )
    target_variable: str = Field(
        default="tiempo_real_min",
        description="Variable predicha"
    )
    input_features: Dict[str, Any]
    rmse_validation: Optional[float] = Field(
        default=None,
        description="RMSE del modelo en validación"
    )
    status: str = Field(
        default="success",
        description="Estado de la predicción"
    )


# ======================================================================
# SCHEMAS DE MÉTRICAS
# ======================================================================

class MetricaFlashcard(BaseModel):
    """Schema para una métrica individual en flashcard."""
    
    label: str = Field(
        ...,
        description="Etiqueta de la métrica (ej: ETA Promedio)"
    )
    
    value: float = Field(
        ...,
        description="Valor numérico"
    )
    
    unit: str = Field(
        default="min",
        description="Unidad (min, %, etc.)"
    )
    
    trend: Optional[str] = Field(
        default=None,
        description="Tendencia (up, down, neutral)"
    )


class FlashcardsOutput(BaseModel):
    """Schema para respuesta de flashcards (KPIs)."""
    
    timestamp: datetime
    model_name: str
    n_records: int = Field(
        description="Número de registros analizados"
    )
    flashcards: List[MetricaFlashcard]
    
    class Config:
        example = {
            "timestamp": "2026-04-12T22:00:00",
            "model_name": "linear",
            "n_records": 10517,
            "flashcards": [
                {
                    "label": "ETA Promedio",
                    "value": 25.5,
                    "unit": "min",
                    "trend": None
                },
                {
                    "label": "ETA Máximo",
                    "value": 89.3,
                    "unit": "min",
                    "trend": None
                },
                {
                    "label": "ETA Mínimo",
                    "value": 5.2,
                    "unit": "min",
                    "trend": None
                },
                {
                    "label": "Error Promedio (MAE)",
                    "value": 3.98,
                    "unit": "min",
                    "trend": "down"
                }
            ]
        }


class ModelMetrics(BaseModel):
    """Schema para métricas de un modelo."""
    
    model_name: str
    mae: float
    mse: float
    rmse: float
    mape: float
    n_train: int
    n_validation: int


class ModelsMetricsOutput(BaseModel):
    """Schema para respuesta con métricas de todos los modelos."""
    
    timestamp: datetime
    models: List[ModelMetrics]
    best_model: str = Field(
        description="Modelo con menor RMSE"
    )
    
    class Config:
        example = {
            "timestamp": "2026-04-12T22:00:00",
            "models": [],
            "best_model": "linear"
        }


# ======================================================================
# SCHEMAS DE HEALTH CHECK
# ======================================================================

class HealthOutput(BaseModel):
    """Schema para respuesta de health check."""
    
    status: str = Field(
        default="ok",
        description="Estado del servicio (ok, error)"
    )
    
    timestamp: datetime
    
    models_loaded: List[str] = Field(
        description="Modelos cargados exitosamente"
    )
    
    models_failed: List[str] = Field(
        default_factory=list,
        description="Modelos que fallaron al cargar"
    )
    
    metrics_available: bool = Field(
        description="Archivo de métricas disponible"
    )


# ======================================================================
# SCHEMAS DE ERROR
# ======================================================================

class ErrorOutput(BaseModel):
    """Schema para respuestas de error."""
    
    detail: str = Field(
        description="Descripción del error"
    )
    
    error_code: str = Field(
        description="Código de error"
    )
    
    timestamp: datetime
