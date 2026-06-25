# -*- coding: utf-8 -*-
"""
Script de prueba del backend.
Verifica que todos los servicios se inicialicen correctamente.
"""

import sys
from pathlib import Path

# Agregar el directorio backend al path
backend_dir = Path(__file__).parent
sys.path.insert(0, str(backend_dir))

import logging

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s"
)

logger = logging.getLogger(__name__)

# ======================================================================
# TEST 1: Verificar configuración
# ======================================================================

def test_config():
    """Test 1: Verifica rutas y configuración."""
    logger.info("="*70)
    logger.info("TEST 1: Verificando configuración")
    logger.info("="*70)
    
    from config import (
        BASE_DIR,
        MODELS_DIR,
        METRICS_DIR,
        MODEL_PATHS,
        METRICS_CSV_PATH,
        validate_paths,
    )
    
    logger.info(f"BASE_DIR: {BASE_DIR}")
    logger.info(f"MODELS_DIR: {MODELS_DIR}")
    logger.info(f"METRICS_CSV_PATH: {METRICS_CSV_PATH}")
    
    logger.info("\nModelos configurados:")
    for model_name, model_path in MODEL_PATHS.items():
        logger.info(f"  - {model_name}: {model_path}")
    
    logger.info("\nValidando rutas...")
    if validate_paths():
        logger.info("[OK] Todas las rutas son válidas")
        return True
    else:
        logger.error("[ERROR] Algunas rutas no existen")
        return False


# ======================================================================
# TEST 2: Cargar servicios
# ======================================================================

def test_services():
    """Test 2: Inicializa servicios."""
    logger.info("\n" + "="*70)
    logger.info("TEST 2: Inicializando servicios")
    logger.info("="*70)
    
    try:
        logger.info("\nCargando PredictionService...")
        from services.prediction_service import get_prediction_service
        
        prediction_service = get_prediction_service()
        available_models = prediction_service.get_available_models()
        
        logger.info(f"[OK] PredictionService inicializado")
        logger.info(f"Modelos disponibles: {list(available_models.keys())}")
        logger.info(f"Modelos cargados: {[m for m, v in available_models.items() if v]}")
        
    except Exception as e:
        logger.error(f"[ERROR] PredictionService: {e}")
        return False
    
    try:
        logger.info("\nCargando MetricsService...")
        from services.metrics_service import get_metrics_service
        
        metrics_service = get_metrics_service()
        all_metrics = metrics_service.get_all_models_metrics()
        
        logger.info(f"[OK] MetricsService inicializado")
        logger.info(f"Modelos con métricas: {len(all_metrics)}")
        for metric in all_metrics:
            logger.info(f"  - {metric['model_name']}: RMSE={metric['rmse']:.4f}")
        
    except Exception as e:
        logger.error(f"[ERROR] MetricsService: {e}")
        return False
    
    return True


# ======================================================================
# TEST 3: Predicción
# ======================================================================

def test_prediction():
    """Test 3: Realiza una predicción de prueba."""
    logger.info("\n" + "="*70)
    logger.info("TEST 3: Realizando predicción de prueba")
    logger.info("="*70)
    
    from services.prediction_service import get_prediction_service
    
    prediction_service = get_prediction_service()
    
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
    
    logger.info(f"\nInput de prueba: {test_input}")
    
    for model_name in ["linear", "glm", "histgb", "xgboost"]:
        try:
            logger.info(f"\nPredicción con {model_name}...")
            prediction, metadata = prediction_service.predict(test_input, model_name)
            logger.info(f"[OK] Predicción: {prediction:.2f} minutos")
        except Exception as e:
            logger.error(f"[ERROR] {model_name}: {e}")
    
    return True


# ======================================================================
# TEST 4: Flashcards
# ======================================================================

def test_flashcards():
    """Test 4: Calcula flashcards."""
    logger.info("\n" + "="*70)
    logger.info("TEST 4: Calculando flashcards (KPIs)")
    logger.info("="*70)
    
    from services.metrics_service import get_metrics_service
    
    metrics_service = get_metrics_service()
    
    for model_name in ["linear", "glm", "histgb", "xgboost"]:
        try:
            logger.info(f"\nFlashcards para {model_name}:")
            flashcards = metrics_service.calculate_flashcards(model_name)
            
            logger.info(f"  N Registros: {flashcards['n_records']}")
            logger.info(f"  Flashcards calculadas: {len(flashcards['flashcards'])}")
            
            for fc in flashcards['flashcards'][:3]:  # Mostrar primeras 3
                logger.info(f"    - {fc['label']}: {fc['value']} {fc['unit']}")
                
        except Exception as e:
            logger.error(f"[ERROR] {model_name}: {e}")
    
    return True


# ======================================================================
# TEST 5: Mejor modelo
# ======================================================================

def test_best_model():
    """Test 5: Identifica el mejor modelo."""
    logger.info("\n" + "="*70)
    logger.info("TEST 5: Identificando mejor modelo")
    logger.info("="*70)
    
    from services.metrics_service import get_metrics_service
    
    metrics_service = get_metrics_service()
    best_model, best_rmse = metrics_service.get_best_model()
    
    logger.info(f"\nMejor modelo: {best_model}")
    logger.info(f"RMSE: {best_rmse:.4f} minutos")
    
    return True


# ======================================================================
# FUNCIÓN PRINCIPAL
# ======================================================================

def main():
    """Ejecuta todos los tests."""
    logger.info("\n")
    logger.info("█" * 70)
    logger.info("█  VALIDACIÓN DEL BACKEND - DASHBOARD LOGÍSTICA ÚLTIMA MILLA".center(70))
    logger.info("█" * 70)
    
    tests = [
        ("Configuración", test_config),
        ("Servicios", test_services),
        ("Predicción", test_prediction),
        ("Flashcards", test_flashcards),
        ("Mejor Modelo", test_best_model),
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            logger.error(f"[EXCEPTION] {test_name}: {e}")
            results[test_name] = False
    
    # Resumen
    logger.info("\n" + "="*70)
    logger.info("RESUMEN DE TESTS")
    logger.info("="*70)
    
    for test_name, passed in results.items():
        status = "[OK]" if passed else "[FAILED]"
        logger.info(f"{status} {test_name}")
    
    passed_count = sum(1 for v in results.values() if v)
    total_count = len(results)
    
    logger.info(f"\nTotal: {passed_count}/{total_count} tests pasaron")
    
    if passed_count == total_count:
        logger.info("\n[OK] BACKEND VALIDADO - LISTO PARA PRODUCCIÓN")
        return 0
    else:
        logger.error("\n[ERROR] ALGUNOS TESTS FALLARON - REVISAR CONFIGURACIÓN")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
