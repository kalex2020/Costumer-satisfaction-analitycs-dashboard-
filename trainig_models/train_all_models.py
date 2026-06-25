# ======================================================================
# SCRIPT MAESTRO: ENTRENAMIENTO DE LOS 4 MODELOS
# Orquestación del pipeline: Linear, GLM, HistGB, XGBoost
# ======================================================================

import os
import sys
import subprocess
import time
import pandas as pd
from datetime import datetime

# ======================================================================
# 1. CONFIGURACIÓN
# ======================================================================

BASE_PATH = r"K:\Dashboard Delivery"
TRAINIG_MODELS_PATH = os.path.join(BASE_PATH, "trainig_models")
MODELS_PATH = os.path.join(BASE_PATH, "models")
METRICS_PATH = os.path.join(BASE_PATH, "trainig_models", "metrics")

# Definir scripts a ejecutar en orden
scripts_to_run = [
    ("Linear Regression", os.path.join(TRAINIG_MODELS_PATH, "train_linear_regression.py")),
    ("GLM", os.path.join(TRAINIG_MODELS_PATH, "train_glm.py")),
    ("HistGradientBoosting", os.path.join(TRAINIG_MODELS_PATH, "train_histgb.py")),
    ("XGBoost", os.path.join(TRAINIG_MODELS_PATH, "train_xgboost.py")),
]

METRICS_CSV = os.path.join(METRICS_PATH, "model_metrics.csv")
FINAL_REPORT = os.path.join(METRICS_PATH, "training_report.txt")

# ======================================================================
# 2. FUNCIONES AUXILIARES
# ======================================================================

# -*- coding: utf-8 -*-

def print_banner(text):
    """Imprimir banner decorativo."""
    print("\n" + "=" * 70)
    print(text.center(70))
    print("=" * 70)

def print_step(step_num, step_name):
    """Imprimir paso numerado."""
    print(f"\n[STEP {step_num}] {step_name}")
    print("-" * 70)

def run_script(script_path, script_name):
    """Ejecutar un script Python y capturar salida."""
    try:
        print(f"\n-- Ejecutando: {script_name}")
        print(f"   Path: {script_path}")
        
        # Ejecutar el script sin capturar output (imprime directamente en consola)
        result = subprocess.run(
            [sys.executable, script_path],
            timeout=600  # Timeout de 10 minutos
        )
        
        if result.returncode != 0:
            print(f"\n[ERROR] Error al ejecutar {script_name}")
            return False
        else:
            print(f"[OK] {script_name} completado exitosamente")
            return True
            
    except subprocess.TimeoutExpired:
        print(f"[ERROR] Timeout: {script_name} tardo mas de 10 minutos")
        return False
    except Exception as e:
        print(f"[ERROR] Excepcion en {script_name}: {e}")
        return False

def load_metrics():
    """Cargar métricas desde CSV consolidado."""
    try:
        if os.path.exists(METRICS_CSV):
            metrics_df = pd.read_csv(METRICS_CSV)
            return metrics_df
        else:
            print(f"⚠ No se encontró archivo de métricas en: {METRICS_CSV}")
            return None
    except Exception as e:
        print(f"✗ Error al cargar métricas: {e}")
        return None

def generate_report(metrics_df):
    """Generar reporte de entrenamiento."""
    try:
        report = []
        report.append("=" * 80)
        report.append("REPORTE FINAL DE ENTRENAMIENTO - MODELOS DE REGRESIÓN")
        report.append("Logística de Última Milla - Intrak Bogotá")
        report.append("=" * 80)
        report.append(f"\nFecha de generación: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"\nDataset: tiempo_real_min (minutos de entrega)")
        report.append(f"Objetivo: Predicción con validación temporal sin leakage\n")
        
        report.append("=" * 80)
        report.append("MÉTRICAS POR MODELO (Validación)")
        report.append("=" * 80)
        
        if metrics_df is not None and len(metrics_df) > 0:
            # Convertir a formato más legible
            for idx, row in metrics_df.iterrows():
                report.append(f"\n[{idx + 1}] {row['model_name'].upper()}")
                report.append(f"  MAE (Error Absoluto Medio):     {row['MAE']:.4f} minutos")
                report.append(f"  MSE (Error Cuadrático Medio):   {row['MSE']:.4f}")
                report.append(f"  RMSE (Raíz del MSE):            {row['RMSE']:.4f} minutos")
                report.append(f"  MAPE (Error Absoluto Porcentual): {row['MAPE']:.4f} ({row['MAPE']*100:.2f}%)")
                report.append(f"  Datos entrenamiento:            {int(row['n_train'])} muestras")
                report.append(f"  Datos validación:               {int(row['n_validation'])} muestras")
            
            # Encontrar mejor modelo por RMSE
            best_model_idx = metrics_df['RMSE'].idxmin()
            best_model = metrics_df.loc[best_model_idx]
            
            report.append("\n" + "=" * 80)
            report.append("RECOMENDACIÓN")
            report.append("=" * 80)
            report.append(f"\nMejor modelo (menor RMSE): {best_model['model_name']}")
            report.append(f"Error RMSE: {best_model['RMSE']:.4f} minutos")
            
            # Tabla comparativa
            report.append("\n" + "=" * 80)
            report.append("TABLA COMPARATIVA")
            report.append("=" * 80 + "\n")
            
            # Header
            report.append(f"{'Modelo':<30} {'MAE':<10} {'RMSE':<10} {'MAPE':<10}")
            report.append("-" * 70)
            
            # Filas
            for idx, row in metrics_df.iterrows():
                model_name = row['model_name'][:28]  # Limitar longitud
                report.append(f"{model_name:<30} {row['MAE']:<10.4f} {row['RMSE']:<10.4f} {row['MAPE']:<10.4f}")
            
            report.append("\n" + "=" * 80)
            report.append("MODELOS GUARDADOS")
            report.append("=" * 80)
            
            models_dir = os.path.join(BASE_PATH, "models")
            if os.path.exists(models_dir):
                model_files = [f for f in os.listdir(models_dir) if f.endswith('.joblib')]
                for mf in model_files:
                    file_path = os.path.join(models_dir, mf)
                    file_size = os.path.getsize(file_path) / 1024  # KB
                    report.append(f"\n[OK] {mf:<30} ({file_size:.2f} KB)")
        
        report.append("\n" + "=" * 80)
        report.append("FIN DEL REPORTE")
        report.append("=" * 80)
        
        return "\n".join(report)
        
    except Exception as e:
        return f"Error generando reporte: {e}"

def save_report(report_text, report_path):
    """Guardar reporte en archivo."""
    try:
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
        print(f"\n✓ Reporte guardado en: {report_path}")
        return True
    except Exception as e:
        print(f"✗ Error al guardar reporte: {e}")
        return False

# ======================================================================
# 3. EJECUCIÓN PRINCIPAL
# ======================================================================

def main():
    print_banner("PIPELINE DE ENTRENAMIENTO: 4 MODELOS DE REGRESIÓN")
    
    # Verificar que existan los scripts
    print_step(1, "Verificación de archivos")
    all_scripts_exist = True
    for script_name, script_path in scripts_to_run:
        if os.path.exists(script_path):
            print(f"[OK] Encontrado: {script_name} ({script_path})")
        else:
            print(f"[ERROR] NO ENCONTRADO: {script_name} ({script_path})")
            all_scripts_exist = False
    
    if not all_scripts_exist:
        print("\n[ERROR] Algunos scripts no existen. Abortando.")
        sys.exit(1)
    
    # Crear directorios de destino
    print_step(2, "Creación de directorios de destino")
    os.makedirs(MODELS_PATH, exist_ok=True)
    os.makedirs(METRICS_PATH, exist_ok=True)
    print(f"[OK] Directorio de modelos: {MODELS_PATH}")
    print(f"[OK] Directorio de métricas: {METRICS_PATH}")
    
    # Ejecutar scripts de entrenamiento
    print_step(3, "Entrenamiento de modelos")
    results = {}
    for script_name, script_path in scripts_to_run:
        success = run_script(script_path, script_name)
        results[script_name] = success
        
        if success:
            print(f"[OK] {script_name}: OK")
        else:
            print(f"[ERROR] {script_name}: FALLO")
        
        time.sleep(1)  # Pequeña pausa entre scripts
    
    # Resumen de ejecución
    print_step(4, "Resumen de ejecución")
    successful = sum(1 for v in results.values() if v)
    total = len(results)
    print(f"\nModelos entrenados exitosamente: {successful}/{total}")
    
    for script_name, success in results.items():
        status = "[OK]" if success else "[ERROR]"
        print(f"  {script_name:<30} {status}")
    
    # Cargar y mostrar métricas
    print_step(5, "Consolidación de métricas")
    metrics_df = load_metrics()
    
    if metrics_df is not None:
        print(f"\n[OK] Métricas consolidadas: {len(metrics_df)} modelos")
        print(f"\n{metrics_df.to_string(index=False)}")
    else:
        print("\n[WARNING] No se encontraron métricas consolidadas")
    
    # Generar reporte final
    print_step(6, "Generación de reporte final")
    report_text = generate_report(metrics_df)
    print(report_text)
    
    # Guardar reporte
    if save_report(report_text, FINAL_REPORT):
        print(f"[OK] Reporte guardado en: {FINAL_REPORT}")
    
    # Resumen final
    print_banner("ENTRENAMIENTO COMPLETADO")
    print(f"\nModelos guardados en:      {MODELS_PATH}")
    print(f"Métricas guardadas en:     {METRICS_CSV}")
    print(f"Reporte guardado en:       {FINAL_REPORT}")
    print("\nProximos pasos:")
    print("1. Revisar métricas en: model_metrics.csv")
    print("2. Evaluar desempeño en: training_report.txt")
    print("3. Usar modelos .joblib para predicciones")
    print("4. Integrar en dashboard Power BI")
    print("\n" + "=" * 70)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n[ERROR] Proceso interrumpido por usuario")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] Error crítico: {e}")
        sys.exit(1)
