# -*- coding: utf-8 -*-
"""
Backend FastAPI para Dashboard de Logística Última Milla.
Sirve predicciones y KPIs a un frontend HTML.
"""

import logging
from datetime import datetime
import calendar
from typing import Optional, List

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from config import (
    API_TITLE,
    API_VERSION,
    API_DESCRIPTION,
    DEFAULT_MODEL,
    MODEL_PATHS,
)
from schemas import (
    PredictionInput,
    PredictionOutput,
    FlashcardsOutput,
    ModelsMetricsOutput,
    ModelMetrics,
    HealthOutput,
    MetricaFlashcard,
)
from services.prediction_service import get_prediction_service
from services.metrics_service import get_metrics_service


def _resolve_target_month(month_str: Optional[str]) -> tuple[int, int, str]:
    """Resuelve año/mes objetivo desde YYYY-MM o siguiente mes por defecto."""
    if month_str:
        year, month = month_str.split("-")
        y = int(year)
        m = int(month)
        if m < 1 or m > 12:
            raise ValueError("El parámetro month debe estar entre 01 y 12")
    else:
        now = datetime.now()
        if now.month == 12:
            y, m = now.year + 1, 1
        else:
            y, m = now.year, now.month + 1

    return y, m, f"{y}-{m:02d}"


def _prepare_feature_row(row, year: int, month: int, day: int, dow: int) -> dict:
    """Construye un registro de features válido para inferencia."""
    distancia = float(row.get("distancia_km", 5.0) or 5.0)
    tiempo_estimado = float(row.get("tiempo_estimado_min", 45.0) or 45.0)

    eficiencia = float(row.get("eficiencia_estimacion", 0.0) or 0.0)
    if eficiencia <= 0 and distancia > 0:
        eficiencia = tiempo_estimado / distancia

    return {
        "localidad": str(row.get("localidad", "Kennedy")),
        "tipo_via": str(row.get("tipo_via", "Calle")),
        "tipo_zona": str(row.get("tipo_zona", "Urbana")),
        "vehiculo": str(row.get("vehiculo", "Moto")),
        "estrato": int(row.get("estrato", 3) or 3),
        "distancia_km": distancia,
        "nivel_trafico": float(row.get("nivel_trafico", 2.0) or 2.0),
        "lluvia_mm": float(row.get("lluvia_mm", 0.0) or 0.0),
        "pendiente_promedio": float(row.get("pendiente_promedio", 0.5) or 0.5),
        "zona_riesgo": int(row.get("zona_riesgo", 0) or 0),
        "cliente_ausente": int(row.get("cliente_ausente", 0) or 0),
        "intentos_entrega": int(row.get("intentos_entrega", 1) or 1),
        "tiempo_estimado_min": tiempo_estimado,
        "eficiencia_estimacion": float(eficiencia),
        "anio": int(year),
        "mes": int(month),
        "dia": int(day),
        "dia_semana": int(dow),
    }


def _predict_eta_with_fallback(prediction_service, features: dict, requested_model: str) -> float:
    """Predice ETA intentando modelo solicitado y fallback robusto."""
    models_to_try = [requested_model, "linear", "histgb", "xgboost", "glm"]
    tried = set()

    for model_name in models_to_try:
        if model_name in tried:
            continue
        tried.add(model_name)

        if not prediction_service.validate_model(model_name):
            continue

        try:
            pred, _ = prediction_service.predict(features, model_name)
            return float(pred)
        except Exception:
            continue

    raise RuntimeError("No fue posible obtener predicción con ningún modelo disponible")


def _build_monthly_future_scenario(modelo: str, month_str: Optional[str]) -> dict:
    """Genera proyección mensual usando inferencias del modelo sobre patrones históricos."""
    metrics_service = get_metrics_service()
    prediction_service = get_prediction_service()

    template_df = metrics_service.validation_data.get("histgb")
    if template_df is None or template_df.empty:
        template_df = metrics_service.maestro_data

    if template_df is None or template_df.empty:
        raise ValueError("No hay datos históricos para construir predicciones futuras")

    base_df = template_df.copy()
    if "dia_semana" not in base_df.columns:
        base_df["dia_semana"] = np.arange(len(base_df)) % 7
    if "dia" not in base_df.columns:
        base_df["dia"] = (np.arange(len(base_df)) % 30) + 1

    year, month, horizon = _resolve_target_month(month_str)
    days_in_month = calendar.monthrange(year, month)[1]

    dow_counts = base_df.groupby("dia_semana").size()
    avg_dow_count = max(float(dow_counts.mean()), 1.0)
    dow_factor = {int(k): float(v / avg_dow_count) for k, v in dow_counts.items()}

    if "dia" in base_df.columns:
        demand_base = int(round(len(base_df) / max(base_df["dia"].nunique(), 1)))
    else:
        demand_base = int(round(len(base_df) / 30))
    demand_base = max(demand_base, 40)

    daily_rows = []
    monthly_eta_samples = []
    all_traffic_points = []

    for day in range(1, days_in_month + 1):
        dow = int(datetime(year, month, day).weekday())
        day_pool = base_df[base_df["dia_semana"] == dow]
        if day_pool.empty:
            day_pool = base_df

        day_factor = dow_factor.get(dow, 1.0)
        demand_pred = int(max(20, round(demand_base * day_factor)))

        sample_n = int(min(max(10, demand_pred // 20), 28))
        sample_df = day_pool.sample(n=sample_n, replace=len(day_pool) < sample_n, random_state=year * 10000 + month * 100 + day)

        pred_eta = []
        pred_delay_flag = []
        traffic_vals = []

        for _, row in sample_df.iterrows():
            features = _prepare_feature_row(row, year, month, day, dow)
            eta_pred = _predict_eta_with_fallback(prediction_service, features, modelo)
            pred_eta.append(float(eta_pred))
            pred_delay_flag.append(1 if eta_pred > 60 else 0)
            traffic_vals.append(float(features["nivel_trafico"]))

        eta_mean = float(np.mean(pred_eta)) if pred_eta else 0.0
        otif_day = float(np.mean(np.array(pred_eta) <= 60) * 100) if pred_eta else 0.0
        delay_day = float(100.0 - otif_day)
        traffic_day = float(np.mean(traffic_vals)) if traffic_vals else 0.0
        risk_day = float(delay_day)

        monthly_eta_samples.extend(pred_eta)
        all_traffic_points.append({
            "day": day,
            "traffic": traffic_day,
            "risk": risk_day,
            "demand": demand_pred,
        })

        daily_rows.append({
            "day": day,
            "eta_expected": eta_mean,
            "otif_expected": otif_day,
            "delays_expected": delay_day,
            "demand_predicted": demand_pred,
            "traffic_expected": traffic_day,
            "risk_delay": risk_day,
        })

    daily_df = pd.DataFrame(daily_rows)

    # Riesgo por localidad en horizonte futuro
    risk_by_localidad = []
    if "localidad" in base_df.columns:
        top_localidades = base_df["localidad"].value_counts().head(6).index.tolist()
        for idx, localidad in enumerate(top_localidades):
            local_pool = base_df[base_df["localidad"] == localidad]
            if local_pool.empty:
                continue

            sample_n = int(min(max(8, len(local_pool) // 40), 24))
            local_sample = local_pool.sample(n=sample_n, replace=len(local_pool) < sample_n, random_state=year * 10 + month + idx)

            local_preds = []
            for _, row in local_sample.iterrows():
                features = _prepare_feature_row(row, year, month, 15, 2)
                features["localidad"] = str(localidad)
                eta_pred = _predict_eta_with_fallback(prediction_service, features, modelo)
                local_preds.append(float(eta_pred))

            if local_preds:
                risk_by_localidad.append({
                    "localidad": str(localidad),
                    "riesgo": float(np.mean(np.array(local_preds) > 60) * 100),
                    "eta": float(np.mean(local_preds)),
                })

    historical_source = metrics_service.validation_data.get(modelo.lower())
    if historical_source is None or historical_source.empty:
        historical_source = base_df

    hist_eta = float(historical_source["tiempo_real_min"].mean()) if "tiempo_real_min" in historical_source.columns else float(base_df["tiempo_estimado_min"].mean())
    hist_delays = float(np.mean(historical_source["tiempo_real_min"] > 60) * 100) if "tiempo_real_min" in historical_source.columns else float(np.mean(base_df["tiempo_estimado_min"] > 60) * 100)

    if "dia" in historical_source.columns:
        hist_daily_demand = int(round(len(historical_source) / max(historical_source["dia"].nunique(), 1)))
    else:
        hist_daily_demand = int(round(len(historical_source) / 30))

    summary = {
        "horizon": horizon,
        "month": f"{month:02d}",
        "year": year,
        "eta_expected_mean": float(daily_df["eta_expected"].mean()),
        "otif_expected": float(daily_df["otif_expected"].mean()),
        "delays_expected": float(daily_df["delays_expected"].mean()),
        "demand_month_predicted": int(daily_df["demand_predicted"].sum()),
        "demand_daily_predicted_mean": float(daily_df["demand_predicted"].mean()),
    }

    comparison = [
        {
            "metric": "Demanda diaria",
            "historical": float(hist_daily_demand),
            "predicted": float(daily_df["demand_predicted"].mean()),
        },
        {
            "metric": "ETA promedio (min)",
            "historical": float(hist_eta),
            "predicted": float(daily_df["eta_expected"].mean()),
        },
        {
            "metric": "% Retrasos",
            "historical": float(hist_delays),
            "predicted": float(daily_df["delays_expected"].mean()),
        },
    ]

    return {
        "summary": summary,
        "daily": daily_rows,
        "eta_distribution": monthly_eta_samples,
        "risk_by_localidad": risk_by_localidad,
        "traffic_risk_points": all_traffic_points,
        "historical_vs_predicted": comparison,
    }

# ======================================================================
# CONFIGURACIÓN DE LOGGING
# ======================================================================

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s"
)

logger = logging.getLogger(__name__)

# ======================================================================
# INICIALIZACIÓN DE LA APP
# ======================================================================

app = FastAPI(
    title=API_TITLE,
    version=API_VERSION,
    description=API_DESCRIPTION,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# ======================================================================
# CONFIGURACIÓN DE CORS
# ======================================================================

# Permitir CORS para frontend local
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8080",
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:8080",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
        "file://",  # Para archivos locales
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ======================================================================
# INICIALIZACIÓN DE SERVICIOS
# ======================================================================

# Los servicios se inicializan la primera vez que se llaman (Singleton)
# Esto ocurre en cada endpoint

# ======================================================================
# EVENTOS DE LIFECYCLE
# ======================================================================

@app.on_event("startup")
async def startup_event():
    """Se ejecuta al iniciar la aplicación."""
    logger.info("="*70)
    logger.info("INICIANDO BACKEND DE DASHBOARD")
    logger.info("="*70)
    
    # Precargar servicios
    logger.info("Precargando servicios...")
    prediction_service = get_prediction_service()
    metrics_service = get_metrics_service()
    
    logger.info(f"Modelos disponibles: {list(prediction_service.get_available_models().keys())}")
    logger.info("Backend inicializado correctamente")


@app.on_event("shutdown")
async def shutdown_event():
    """Se ejecuta al apagar la aplicación."""
    logger.info("Cerrando backend...")


# ======================================================================
# ENDPOINTS: HEALTH & STATUS
# ======================================================================

@app.get(
    "/health",
    response_model=HealthOutput,
    summary="Health check del servicio",
    tags=["Health"]
)
async def health_check() -> HealthOutput:
    """
    Verifica el estado del servicio.
    Retorna status, modelos cargados y disponibilidad de datos.
    """
    try:
        prediction_service = get_prediction_service()
        metrics_service = get_metrics_service()
        
        # Obtener modelos disponibles
        available_models = prediction_service.get_available_models()
        models_loaded = [m for m, available in available_models.items() if available]
        models_failed = [m for m, available in available_models.items() if not available]
        
        # Verificar disponibilidad de métricas
        metrics_available = len(metrics_service.model_metrics) > 0
        
        response = HealthOutput(
            status="ok" if len(models_loaded) > 0 else "degraded",
            timestamp=datetime.now(),
            models_loaded=models_loaded,
            models_failed=models_failed,
            metrics_available=metrics_available,
        )
        
        logger.info(f"Health check: {len(models_loaded)} modelos cargados")
        return response
    
    except Exception as e:
        logger.error(f"Error en health check: {str(e)}")
        raise HTTPException(status_code=500, detail="Error en health check")


# ======================================================================
# ENDPOINTS: PREDICCIÓN
# ======================================================================

@app.post(
    "/predict/eta",
    response_model=PredictionOutput,
    summary="Predicción de ETA",
    tags=["Prediction"]
)
async def predict_eta(input_data: PredictionInput) -> PredictionOutput:
    """
    Realiza una predicción de tiempo real de entrega (ETA) en minutos.
    
    Parámetros de entrada:
    - localidad: Localidad de destino
    - distancia_km: Distancia en kilómetros
    - nivel_trafico: Nivel de tráfico (1-5)
    - lluvia_mm: Precipitación en milímetros
    - estrato: Estrato socioeconómico (1-6)
    - tipo_via: Tipo de vía
    - tipo_zona: Tipo de zona (urbana, suburbana)
    - vehiculo: Tipo de vehículo
    - tiempo_estimado_min: Tiempo estimado por algoritmo de ruteo
    - modelo: Modelo a usar (linear, glm, histgb, xgboost)
    
    Retorno:
    - prediction: ETA predicho en minutos
    - model_name: Modelo usado
    - input_features: Features utilizadas
    """
    try:
        # Convertir input a diccionario
        features = input_data.dict(exclude={"modelo"})
        
        # Obtener servicio de predicción
        prediction_service = get_prediction_service()
        
        # Realizar predicción
        prediction_value, metadata = prediction_service.predict(
            features,
            input_data.modelo
        )
        
        # Obtener métricas del modelo para incluir
        metrics_service = get_metrics_service()
        model_metrics = metrics_service.get_model_metrics(input_data.modelo)
        rmse_validation = model_metrics.get("rmse") if model_metrics else None
        
        # Construir respuesta
        response = PredictionOutput(
            timestamp=datetime.now(),
            model_name=input_data.modelo,
            prediction=prediction_value,
            target_variable="tiempo_real_min",
            input_features=features,
            rmse_validation=rmse_validation,
            status="success",
        )
        
        logger.info(f"Predicción exitosa: {prediction_value:.2f} min con modelo {input_data.modelo}")
        return response
    
    except ValueError as e:
        logger.error(f"Error de validación: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error en predicción: {str(e)}")
        raise HTTPException(status_code=500, detail="Error en predicción")


@app.post(
    "/predict/batch",
    summary="Predicciones en batch",
    tags=["Prediction"]
)
async def predict_batch(
    input_data_list: List[PredictionInput],
    modelo: str = Query(DEFAULT_MODEL, description="Modelo a usar")
) -> dict:
    """
    Realiza predicciones múltiples (batch).
    Útil para predecir sobre datasets grandes.
    """
    try:
        prediction_service = get_prediction_service()
        
        # Convertir a diccionarios
        features_list = [item.dict(exclude={"modelo"}) for item in input_data_list]
        
        # Predicciones
        predictions = prediction_service.batch_predict(features_list, modelo)
        
        logger.info(f"Batch predictions completadas: {len(predictions)} registros")
        
        return {
            "timestamp": datetime.now(),
            "model_name": modelo,
            "n_records": len(predictions),
            "predictions": predictions,
            "status": "success",
        }
    
    except Exception as e:
        logger.error(f"Error en batch predict: {str(e)}")
        raise HTTPException(status_code=500, detail="Error en batch predict")


# ======================================================================
# ENDPOINTS: MÉTRICAS & KPIs
# ======================================================================

@app.get(
    "/metrics/flashcards",
    response_model=FlashcardsOutput,
    summary="Flashcards (KPIs) para dashboard",
    tags=["Metrics"]
)
async def get_flashcards(
    modelo: str = Query(DEFAULT_MODEL, description="Modelo a analizar")
) -> FlashcardsOutput:
    """
    Retorna métricas agregadas (flashcards) para mostrar en el dashboard.
    
    Incluye:
    - ETA promedio, máximo, mínimo
    - Error promedio (MAE)
    - RMSE y MAPE del modelo
    - Número de registros analizados
    """
    try:
        metrics_service = get_metrics_service()
        
        # Calcular flashcards
        flashcards_data = metrics_service.calculate_flashcards(modelo)
        
        # Convertir a schema
        flashcards_objs = [
            MetricaFlashcard(**fc) for fc in flashcards_data["flashcards"]
        ]
        
        response = FlashcardsOutput(
            timestamp=flashcards_data["timestamp"],
            model_name=flashcards_data["model_name"],
            n_records=flashcards_data["n_records"],
            flashcards=flashcards_objs,
        )
        
        logger.info(f"Flashcards retornadas para {modelo}: {len(flashcards_objs)} métricas")
        return response
    
    except Exception as e:
        logger.error(f"Error en flashcards: {str(e)}")
        raise HTTPException(status_code=500, detail="Error al calcular flashcards")


@app.get(
    "/metrics/models",
    response_model=ModelsMetricsOutput,
    summary="Métricas de todos los modelos",
    tags=["Metrics"]
)
async def get_models_metrics() -> ModelsMetricsOutput:
    """
    Retorna métricas de validación (MAE, MSE, RMSE, MAPE) para todos los modelos.
    Útil para comparativas en el dashboard.
    """
    try:
        metrics_service = get_metrics_service()
        
        # Obtener métricas de todos los modelos
        all_metrics = metrics_service.get_all_models_metrics()
        
        # Obtener mejor modelo
        best_model, _ = metrics_service.get_best_model()
        
        # Convertir a schema
        metrics_objs = [
            ModelMetrics(
                model_name=m["model_name"],
                mae=m.get("mae", 0.0),
                mse=m.get("mse", 0.0),
                rmse=m.get("rmse", 0.0),
                mape=m.get("mape", 0.0),
                n_train=m.get("n_train", 0),
                n_validation=m.get("n_validation", 0),
            )
            for m in all_metrics
        ]
        
        response = ModelsMetricsOutput(
            timestamp=datetime.now(),
            models=metrics_objs,
            best_model=best_model,
        )
        
        logger.info(f"Métricas de modelos retornadas: {len(metrics_objs)} modelos")
        return response
    
    except Exception as e:
        logger.error(f"Error en métricas de modelos: {str(e)}")
        raise HTTPException(status_code=500, detail="Error al obtener métricas")


# ======================================================================
# ENDPOINTS: INFO & CONFIG
# ======================================================================

@app.get(
    "/config/models",
    summary="Información de modelos disponibles",
    tags=["Config"]
)
async def get_available_models():
    """Retorna lista de modelos disponibles."""
    try:
        prediction_service = get_prediction_service()
        available = prediction_service.get_available_models()
        
        return {
            "timestamp": datetime.now(),
            "default_model": DEFAULT_MODEL,
            "models": available,
            "models_available": list(available.keys()),
        }
    
    except Exception as e:
        logger.error(f"Error en config/models: {str(e)}")
        raise HTTPException(status_code=500, detail="Error")


@app.get(
    "/metrics/kpis/actual",
    summary="KPIs simplificados para dashboard actual",
    tags=["Metrics"]
)
async def get_kpis_actual(
    modelo: str = Query(DEFAULT_MODEL, description="Modelo a analizar")
):
    """
    Retorna KPIs en formato simplificado para el frontend (Análisis Actual).
    """
    try:
        metrics_service = get_metrics_service()
        val_data = metrics_service.validation_data.get(modelo.lower())
        model_metrics = metrics_service.get_model_metrics(modelo)
        
        kpis = {
            "timestamp": datetime.now().isoformat(),
            "model_name": modelo,
        }
        
        if val_data is not None and "tiempo_real_min" in val_data.columns:
            y = val_data["tiempo_real_min"].values
            kpis["eta_mean"] = float(np.mean(y))
            kpis["eta_max"] = float(np.max(y))
            kpis["eta_min"] = float(np.min(y))
            kpis["eta_std"] = float(np.std(y))
            kpis["n_records"] = len(y)
            
            otif_threshold = 60.0
            otif = (len(y[y <= otif_threshold]) / len(y)) * 100
            kpis["otif"] = float(otif)
            
            delays_pct = ((len(y[y > otif_threshold])) / len(y)) * 100
            kpis["delays_pct"] = float(delays_pct)
        else:
            kpis["eta_mean"] = kpis["eta_max"] = kpis["eta_min"] = kpis["eta_std"] = 0.0
            kpis["n_records"] = 0
            kpis["otif"] = kpis["delays_pct"] = 0.0
        
        if model_metrics:
            kpis["mae"] = float(model_metrics.get("mae", 0.0))
            kpis["rmse"] = float(model_metrics.get("rmse", 0.0))
            kpis["mape"] = float(model_metrics.get("mape", 0.0))
        else:
            kpis["mae"] = kpis["rmse"] = kpis["mape"] = 0.0
        
        return kpis
    except Exception as e:
        logger.error(f"Error en KPIs actuales: {str(e)}")
        raise HTTPException(status_code=500, detail="Error al calcular KPIs")

@app.get(
    "/metrics/kpis/future",
    summary="KPIs simplificados para dashboard futuro",
    tags=["Metrics"]
)
async def get_kpis_future(
    modelo: str = Query(DEFAULT_MODEL, description="Modelo a analizar"),
    month: Optional[str] = Query(None, description="Mes objetivo en formato YYYY-MM")
):
    """
    Retorna KPIs en formato simplificado para el frontend (Análisis Futuro).
    Usa las predicciones para generar los KPIs.
    """
    try:
        scenario = _build_monthly_future_scenario(modelo, month)
        summary = scenario["summary"]
        metrics_service = get_metrics_service()
        model_metrics = metrics_service.get_model_metrics(modelo)
        
        kpis = {
            "timestamp": datetime.now().isoformat(),
            "model_name": modelo,
        }
        
        kpis["eta_mean"] = float(summary["eta_expected_mean"])
        kpis["otif"] = float(summary["otif_expected"])
        kpis["delays_pct"] = float(summary["delays_expected"])
        kpis["n_records"] = int(summary["demand_month_predicted"])
        kpis["horizon"] = summary["horizon"]
        kpis["demand_month_predicted"] = int(summary["demand_month_predicted"])
        kpis["demand_daily_predicted_mean"] = float(summary["demand_daily_predicted_mean"])
        
        if model_metrics:
            kpis["mae"] = float(model_metrics.get("mae", 0.0))
            kpis["rmse"] = float(model_metrics.get("rmse", 0.0))
            kpis["mape"] = float(model_metrics.get("mape", 0.0))
        else:
            kpis["mae"] = kpis["rmse"] = kpis["mape"] = 0.0
        
        return kpis
    except Exception as e:
        logger.error(f"Error en KPIs futuros: {str(e)}")
        raise HTTPException(status_code=500, detail="Error al calcular KPIs futuros")


@app.get(
    "/metrics/kpis/confidence",
    summary="KPIs de confianza global de modelos",
    tags=["Metrics"]
)
async def get_kpis_confidence():
    """Retorna KPIs globales de confianza con promedio sobre los 4 modelos."""
    try:
        metrics_service = get_metrics_service()
        all_metrics = metrics_service.get_all_models_metrics()

        if not all_metrics:
            return {
                "timestamp": datetime.now().isoformat(),
                "n_models": 0,
                "precision_global": 0.0,
                "mae_global": 0.0,
                "rmse_global": 0.0,
                "mape_global": 0.0,
                "best_model": "N/A",
                "best_rmse": 0.0,
                "score_confianza": 0.0,
            }

        mae_values = np.array([float(m.get("mae", 0.0)) for m in all_metrics], dtype=float)
        rmse_values = np.array([float(m.get("rmse", 0.0)) for m in all_metrics], dtype=float)
        mape_values = np.array([float(m.get("mape", 0.0)) * 100 for m in all_metrics], dtype=float)

        mae_global = float(np.mean(mae_values))
        rmse_global = float(np.mean(rmse_values))
        mape_global = float(np.mean(mape_values))
        precision_global = float(max(0.0, 100.0 - mape_global))

        rmse_mean = float(np.mean(rmse_values)) if len(rmse_values) > 0 else 0.0
        rmse_std = float(np.std(rmse_values)) if len(rmse_values) > 0 else 0.0
        variability_penalty = (rmse_std / rmse_mean * 100.0) if rmse_mean > 0 else 100.0
        score_confianza = float(max(0.0, min(100.0, precision_global - variability_penalty * 0.35)))

        best_model, best_rmse = metrics_service.get_best_model()

        return {
            "timestamp": datetime.now().isoformat(),
            "n_models": len(all_metrics),
            "precision_global": precision_global,
            "mae_global": mae_global,
            "rmse_global": rmse_global,
            "mape_global": mape_global,
            "best_model": best_model,
            "best_rmse": float(best_rmse),
            "score_confianza": score_confianza,
        }

    except Exception as e:
        logger.error(f"Error en KPIs de confianza: {str(e)}")
        raise HTTPException(status_code=500, detail="Error al calcular KPIs de confianza")


@app.get(
    "/analytics/actual",
    summary="Datos para Análisis Actual",
    tags=["Analytics"]
)
async def get_analytics_actual(
    modelo: str = Query(DEFAULT_MODEL, description="Modelo a analizar")
):
    """Retorna datos agregados para visualizaciones del análisis actual."""
    try:
        metrics_service = get_metrics_service()
        val_data = metrics_service.validation_data.get(modelo.lower())
        maestro_data = metrics_service.maestro_data
        
        analytics = {}
        
        if val_data is not None:
            # Datos por localidad
            if 'localidad' in val_data.columns and 'tiempo_real_min' in val_data.columns:
                by_localidad = val_data.groupby('localidad')['tiempo_real_min'].agg(['mean', 'count']).reset_index()
                analytics['por_localidad'] = by_localidad.to_dict('records')
            
            # Datos por tipo de vía
            if 'tipo_via' in val_data.columns and 'tiempo_real_min' in val_data.columns:
                by_via = val_data.groupby('tipo_via')['tiempo_real_min'].agg(['mean', 'count']).reset_index()
                analytics['por_via'] = by_via.to_dict('records')
            
            # Evolución temporal (agregar por rangos)
            if 'tiempo_estimado_min' in val_data.columns and 'tiempo_real_min' in val_data.columns:
                # Tomar primeros 20 registros ordenados por tiempo estimado
                sorted_data = val_data.sort_values('tiempo_estimado_min').head(20)
                analytics['evolucion_temporal'] = {
                    'x': list(range(1, len(sorted_data)+1)),
                    'real': sorted_data['tiempo_real_min'].tolist(),
                    'estimado': sorted_data['tiempo_estimado_min'].tolist()
                }
            
            # Impacto del tráfico
            if 'nivel_trafico' in val_data.columns and 'tiempo_real_min' in val_data.columns:
                traffic_impact = val_data.groupby('nivel_trafico')['tiempo_real_min'].agg(['mean', 'count']).reset_index()
                analytics['trafico_impacto'] = traffic_impact.to_dict('records')
            
            # Impacto de lluvia
            if 'lluvia_mm' in val_data.columns and 'tiempo_real_min' in val_data.columns:
                rain_data = val_data[['lluvia_mm', 'tiempo_real_min']].dropna().head(50)
                analytics['lluvia_impacto'] = {
                    'lluvia': rain_data['lluvia_mm'].tolist(),
                    'eta': rain_data['tiempo_real_min'].tolist()
                }
        
        return {"timestamp": datetime.now().isoformat(), "modelo": modelo, "data": analytics}
    
    except Exception as e:
        logger.error(f"Error en analytics/actual: {str(e)}")
        return {"timestamp": datetime.now().isoformat(), "modelo": modelo, "data": {}}


@app.get(
    "/analytics/future",
    summary="Datos para Análisis Futuro (predicciones)",
    tags=["Analytics"]
)
async def get_analytics_future(
    modelo: str = Query(DEFAULT_MODEL, description="Modelo a analizar"),
    month: Optional[str] = Query(None, description="Mes objetivo en formato YYYY-MM")
):
    """Retorna predicciones futuras del modelo."""
    try:
        scenario = _build_monthly_future_scenario(modelo, month)
        daily_df = pd.DataFrame(scenario["daily"])

        predictions = {
            "horizon": scenario["summary"]["horizon"],
            "demanda_diaria": {
                "x": daily_df["day"].astype(int).tolist(),
                "y": daily_df["demand_predicted"].astype(float).tolist(),
            },
            "eta_esperado_diario": {
                "x": daily_df["day"].astype(int).tolist(),
                "y": daily_df["eta_expected"].astype(float).tolist(),
            },
            "otif_esperado_diario": {
                "x": daily_df["day"].astype(int).tolist(),
                "y": daily_df["otif_expected"].astype(float).tolist(),
            },
            "retrasos_esperados_diario": {
                "x": daily_df["day"].astype(int).tolist(),
                "y": daily_df["delays_expected"].astype(float).tolist(),
            },
            "distribucion_eta": {
                "etas": scenario["eta_distribution"],
            },
            "riesgo_por_localidad": scenario["risk_by_localidad"],
            "trafico_vs_riesgo": scenario["traffic_risk_points"],
            "historico_vs_predicho": scenario["historical_vs_predicted"],
        }

        return {"timestamp": datetime.now().isoformat(), "modelo": modelo, "predicciones": predictions}
    
    except Exception as e:
        logger.error(f"Error en analytics/future: {str(e)}")
        return {"timestamp": datetime.now().isoformat(), "modelo": modelo, "predicciones": {}}


@app.get(
    "/predictions/monthly",
    summary="Predicciones mensuales del próximo horizonte",
    tags=["Prediction"]
)
async def get_monthly_predictions(
    modelo: str = Query(DEFAULT_MODEL, description="Modelo a analizar"),
    month: Optional[str] = Query(None, description="Mes objetivo en formato YYYY-MM")
):
    """Retorna escenario mensual completo con demanda, ETA, OTIF y riesgo."""
    try:
        scenario = _build_monthly_future_scenario(modelo, month)
        return {
            "timestamp": datetime.now().isoformat(),
            "modelo": modelo,
            "scenario": scenario,
        }
    except Exception as e:
        logger.error(f"Error en predictions/monthly: {str(e)}")
        raise HTTPException(status_code=500, detail="Error al generar predicciones mensuales")


@app.get(
    "/analytics/confidence",
    summary="Métricas de confianza y calidad del modelo",
    tags=["Analytics"]
)
async def get_analytics_confidence(
    modelo: str = Query(DEFAULT_MODEL, description="Modelo a analizar")
):
    """Retorna métricas de precisión, confianza y evolución del error."""
    try:
        metrics_service = get_metrics_service()
        all_models_metrics = metrics_service.get_all_models_metrics()
        val_data = metrics_service.validation_data.get(modelo.lower()) if modelo else None
        
        confidence = {}
        
        # Comparación de todos los modelos
        models_comparison = []
        for m in all_models_metrics:
            models_comparison.append({
                'modelo': m['model_name'],
                'mae': float(m.get('mae', 0.0)),
                'rmse': float(m.get('rmse', 0.0)),
                'mape': float(m.get('mape', 0.0))
            })
        confidence['modelos_metricas'] = models_comparison
        
        # Evolución del error en el tiempo
        if val_data is not None:
            if 'tiempo_real_min' in val_data.columns:
                errors = (val_data['tiempo_real_min'] - val_data['tiempo_estimado_min']).abs().values
                # Agrupar en 10 segmentos
                segment_size = max(1, len(errors) // 10)
                evolucion = []
                for i in range(0, len(errors), segment_size):
                    segment_error = float(errors[i:i+segment_size].mean())
                    evolucion.append(segment_error)
                confidence['evolucion_error'] = evolucion[:10]
        
        # Distribución del error
        if val_data is not None:
            if 'tiempo_real_min' in val_data.columns and 'tiempo_estimado_min' in val_data.columns:
                errors = (val_data['tiempo_real_min'] - val_data['tiempo_estimado_min']).values
                confidence['distribucion_error'] = errors.tolist()[:200]
        
        return {"timestamp": datetime.now().isoformat(), "modelo": modelo, "confianza": confidence}
    
    except Exception as e:
        logger.error(f"Error en analytics/confidence: {str(e)}")
        return {"timestamp": datetime.now().isoformat(), "modelo": modelo, "confianza": {}}


# ======================================================================
# MANEJO DE ERRORES GLOBAL
# ======================================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Maneja excepciones HTTP."""
    logger.error(f"HTTPException: {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "detail": exc.detail,
            "error_code": "HTTP_ERROR",
            "timestamp": datetime.now().isoformat(),
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Maneja excepciones generales."""
    logger.error(f"General Exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Error interno en el servidor",
            "error_code": "INTERNAL_ERROR",
            "timestamp": datetime.now().isoformat(),
        }
    )


# ======================================================================
# ROOT & DOCUMENTATION
# ======================================================================

@app.get(
    "/",
    tags=["Info"]
)
async def root():
    """Información del API."""
    return {
        "title": API_TITLE,
        "version": API_VERSION,
        "description": API_DESCRIPTION,
        "docs": "/docs",
        "redoc": "/redoc",
        "endpoints": {
            "health": "/health",
            "predict": "/predict/eta",
            "metrics": "/metrics/flashcards",
            "models_metrics": "/metrics/models",
            "config": "/config/models",
        }
    }


# ======================================================================
# PUNTO DE ENTRADA
# ======================================================================

if __name__ == "__main__":
    import uvicorn
    
    # Configurar y ejecutar servidor
    logger.info("Iniciando servidor FastAPI...")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
        access_log=True,
    )
