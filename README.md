# Dashboard de Logística de Última Milla

Proyecto de analítica predictiva para Intrak Bogotá, orientado a medir, anticipar y comunicar el desempeño operativo de entregas de última milla mediante un flujo completo de datos, entrenamiento, API y dashboard web.

## Resumen Ejecutivo

El proyecto convierte un dataset logístico en una solución útil para negocio. Primero se prepara la información histórica, luego se entrenan cuatro modelos de regresión para predecir el tiempo real de entrega y, finalmente, el resultado se expone en un backend Python consumido por un frontend HTML, CSS y JavaScript.

La propuesta combina análisis descriptivo y predictivo. El componente descriptivo explica qué ocurrió en la operación, mientras que el componente predictivo estima qué puede ocurrir con base en variables operativas, geográficas y climáticas. El resultado final es un dashboard que permite lectura gerencial, seguimiento operativo y soporte para decisiones a partir de KPIs como OTIF, porcentaje de retrasos, ETA y comparativos por modelo.

## Alcance Del Proyecto

Este repositorio concentra cuatro capas principales:

1. Preparación de datos con control temporal y sin fuga de información.
2. Entrenamiento y evaluación de modelos predictivos.
3. Exposición de métricas y predicciones mediante una API en Python.
4. Visualización de KPIs y escenarios en un frontend web.

## Flujo Del Proyecto

El flujo implementado fue el siguiente:

1. Se parte del archivo maestro [data/processed/dataset_maestro_intrak_bogota.csv](data/processed/dataset_maestro_intrak_bogota.csv).
2. Se generan particiones de entrenamiento y validación con corte temporal.
3. Se construyen datasets específicos para cada familia de modelo.
4. Se entrenan cuatro modelos de regresión: Linear Regression, GLM, HistGradientBoosting y XGBoost.
5. Se evalúan las métricas en validación y se consolidan en [trainig_models/metrics/model_metrics.csv](trainig_models/metrics/model_metrics.csv).
6. Se guardan modelos y preprocesadores en [models/](models/).
7. El backend en [backend/main.py](backend/main.py) expone métricas, salud del servicio y predicciones.
8. El frontend en [frontend/index.html](frontend/index.html) consume la API y presenta la información en forma de dashboard.

## Contexto Del Problema

El caso de negocio está centrado en la logística de última milla en Bogotá D.C., donde el tiempo de entrega depende de condiciones que cambian rápidamente: tráfico urbano, clima, distancia, zona, tipo de vía y tipo de vehículo. En este entorno, una predicción robusta del ETA ayuda a priorizar rutas, explicar desviaciones y mejorar la planeación operativa.

## Datos Utilizados

El dataset maestro integra variables temporales, geográficas, operativas y de contexto. Las variables más relevantes son:

- Temporales: fecha de pedido, año, mes, día y día de semana.
- Geográficas: localidad y estrato.
- Operativas: tipo de vía, vehículo y distancia en kilómetros.
- Contextuales: nivel de tráfico, lluvia y pendiente promedio.
- Riesgo: zona de riesgo y cliente ausente.
- Planeación: tiempo estimado en minutos.
- Objetivo: tiempo real de entrega en minutos, usado como variable objetivo.

La variable objetivo del proyecto es `tiempo_real_min`.

## Preparación Y Modelado

La preparación de datos se diseñó para evitar sesgos y garantizar consistencia entre entrenamiento e inferencia:

- Se utilizó partición temporal, no aleatoria.
- Se evitó usar información futura en la construcción de variables.
- Se mantuvo separación entre datasets por modelo.
- Se aplicó feature engineering orientado a la operación.

Un indicador derivado relevante fue la eficiencia de estimación:

`eficiencia_estimacion = tiempo_estimado_min / distancia_km`

### Modelos Entrenados

| Modelo | Tipo | Preparación |
| --- | --- | --- |
| Linear Regression | Lineal | OneHot + StandardScaler |
| GLM (Gamma, log-link) | Lineal | OneHot + StandardScaler |
| HistGradientBoosting | Árboles | Encoding ordinal |
| XGBoost | Árboles | Encoding ordinal |

Cada familia de modelos trabaja con su propio preprocesamiento, por lo que no existe un único dataset universal para todos los algoritmos.

## Resultados De Validación

Las métricas consolidadas en validación fueron las siguientes:

| Modelo | MAE | RMSE | MAPE |
| --- | --- | --- | --- |
| Linear Regression | 3.9825 | 4.9766 | 7.09% |
| GLM (Gamma, log-link) | 4.1623 | 5.2545 | 7.38% |
| HistGradientBoosting | 4.0213 | 5.0315 | 7.15% |
| XGBoost | 4.1046 | 5.1345 | 7.28% |

El mejor resultado en RMSE fue Linear Regression, con 4.9766 minutos. En los cuatro modelos el error porcentual se mantuvo alrededor del 7%, lo que indica un comportamiento estable para uso analítico y operativo.

## Conclusiones

- El modelo lineal fue el más competitivo en validación y además ofrece mayor simplicidad de interpretación.
- HistGradientBoosting quedó muy cerca del mejor resultado, por lo que puede considerarse una alternativa robusta si se busca balance entre precisión y flexibilidad.
- GLM y XGBoost también entregan resultados consistentes, pero sin superar al baseline lineal en este caso.
- La diferencia entre modelos no es grande, lo que sugiere que las variables actuales capturan buena parte de la variabilidad del proceso logístico.
- Desde negocio, el sistema ya permite pasar de una lectura histórica a una lectura anticipada del ETA, con soporte para seguimiento por localidad, tráfico, clima y tipo de vía.
- La arquitectura separa entrenamiento, inferencia y visualización, lo que facilita mantenimiento, escalabilidad y nuevas iteraciones del modelo.

## Arquitectura General

```text
Dataset maestro
      ↓
Preparación de datos en Python
      ↓
Entrenamiento de 4 modelos
      ↓
Evaluación y selección del mejor modelo
      ↓
Persistencia de modelos y métricas
      ↓
Backend FastAPI
      ↓
Frontend HTML + CSS + JavaScript
```

## Estructura Del Repositorio

```text
Dashboard Delivery/
├── backend/
│   ├── main.py
│   ├── config.py
│   ├── schemas.py
│   └── services/
├── data/
│   ├── raw/
│   └── processed/
├── frontend/
│   ├── index.html
│   ├── main.js
│   ├── server.py
│   └── styles.css
├── models/
│   ├── glm_model.joblib
│   ├── histgb_model.joblib
│   ├── linear_model.joblib
│   ├── xgboost_model.joblib
│   └── preprocessors/
├── Scripts/
│   ├── prepare_dataset.py
│   ├── simulacion_data.py
│   └── analisis.ipynb
├── trainig_models/
│   ├── train_all_models.py
│   ├── train_glm.py
│   ├── train_histgb.py
│   ├── train_linear_regression.py
│   ├── train_xgboost.py
│   └── metrics/
└── README.md
```

## Backend Y Frontend

El backend en Python centraliza la inferencia, la lectura de métricas y la exposición de endpoints para el dashboard. El frontend consume esos datos y presenta indicadores operativos como OTIF, porcentaje de retrasos, comparativos entre modelos y predicciones individuales.

La documentación específica del backend y del frontend se encuentra en:

- [backend/README.md](backend/README.md)
- [frontend/README.md](frontend/README.md)

## Ejecutar El Proyecto

### Backend

```bash
cd backend
pip install -r requirements.txt
python main.py
```

### Frontend

```bash
cd frontend
python server.py
```

## Entregable Final

El proyecto deja una base funcional para seguimiento y predicción de entregas en última milla. El valor principal no está solo en el modelo ganador, sino en el flujo completo: datos preparados correctamente, entrenamiento trazable, métricas comparables, KPIs como OTIF y porcentaje de retrasos, API lista para consumo y dashboard orientado a negocio.

## Próximos Pasos Recomendados

1. Incorporar monitoreo de desempeño en producción para detectar deriva.
2. Publicar endpoints agregados de análisis futuro y KPIs consolidados.
3. Añadir pruebas automáticas para inferencia y contratos de API.
4. Extender la visualización con escenarios por localidad, clima y horario.
