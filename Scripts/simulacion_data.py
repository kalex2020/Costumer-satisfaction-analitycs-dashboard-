# ==========================================
# INTRAK BOGOTÁ - DATASET SIMULATOR
# Última Milla Urbana
# ==========================================

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

# -----------------------------
# 1. LOAD ORIGINAL DATASET
# -----------------------------

input_path = r"data\raw\Delivery_Logistics.csv"
df = pd.read_csv(input_path)

# -----------------------------
# 2. CONFIG BOGOTÁ PARAMETERS
# -----------------------------

np.random.seed(42)

localidades = [
    ("Suba",0.16), ("Kennedy",0.15), ("Engativá",0.12),
    ("Bosa",0.10), ("Usaquén",0.08), ("Fontibón",0.07),
    ("Chapinero",0.06), ("Teusaquillo",0.05),
    ("Puente Aranda",0.05), ("Tunjuelito",0.04),
    ("Ciudad Bolívar",0.04), ("Barrios Unidos",0.03),
    ("San Cristóbal",0.03), ("Usme",0.02)
]

tipos_via = ["Troncal","Arterial","Local","Destapada"]

tipo_zona = ["Residencial","Industrial","Comercial"]

vehiculos_intrak = [
    "Moto", "Van", "Bicicleta",
    "Camioneta", "Furgón"
]

# -----------------------------
# 3. GENERATE DATE RANGE (5 YEARS)
# -----------------------------

start = datetime(2020,1,1)
end   = datetime(2025,12,31)

date_rng = pd.date_range(start,end,freq="H")

n = len(date_rng)

# -----------------------------
# 4. SIMULATE CORE VARIABLES
# -----------------------------

data = pd.DataFrame()

data["fecha_pedido"] = np.random.choice(date_rng, n)

data["localidad"] = np.random.choice(
    [l[0] for l in localidades],
    n,
    p=[l[1] for l in localidades]
)

data["estrato"] = np.clip(
    np.round(np.random.normal(3,1,n)),1,6
).astype(int)

data["tipo_via"] = np.random.choice(tipos_via,n,
    p=[0.2,0.3,0.45,0.05]
)

data["tipo_zona"] = np.random.choice(tipo_zona,n)

data["vehiculo"] = np.random.choice(vehiculos_intrak,n)

data["distancia_km"] = np.random.gamma(3,1,n).clip(0.5,15)

# -----------------------------
# 5. TRAFFIC SIMULATION
# -----------------------------

hora = data["fecha_pedido"].dt.hour

trafico = []

for h in hora:
    if 6 <= h <=9 or 16<=h<=20:
        trafico.append(
            np.clip(np.random.normal(0.85,0.07),0.5,1)
        )
    else:
        trafico.append(
            np.clip(np.random.normal(0.45,0.05),0.2,0.7)
        )

data["nivel_trafico"] = trafico

# -----------------------------
# 6. RAIN SIMULATION BOGOTÁ
# -----------------------------

mes = data["fecha_pedido"].dt.month

lluvia = []

for m in mes:
    if m in [4,5,10,11]:
        lluvia.append(
            np.random.gamma(2,3)
        )
    else:
        lluvia.append(
            np.abs(np.random.normal(0.2,0.1))
        )

data["lluvia_mm"] = lluvia

# -----------------------------
# 7. TOPOGRAPHY
# -----------------------------

data["pendiente_promedio"] = \
np.abs(np.random.normal(4,2,n)).clip(0,15)

data["zona_riesgo"] = \
np.clip(np.random.normal(0.4,0.2,n),0,1)

data["cliente_ausente"] = \
np.random.binomial(1,0.12,n)

data["intentos_entrega"] = \
np.random.choice([1,2,3],n,p=[0.75,0.2,0.05])

# -----------------------------
# 8. ETA MODEL (REALISTIC)
# -----------------------------

via_factor = {
"Troncal":0,
"Arterial":5,
"Local":9,
"Destapada":20
}

data["tiempo_estimado_min"] = \
20 + data["distancia_km"]*2.5 \
+ data["nivel_trafico"]*20

ruido = np.random.normal(0,5,n)

data["tiempo_real_min"] = \
20 \
+ data["distancia_km"]*2.5 \
+ data["nivel_trafico"]*30 \
+ data["lluvia_mm"]*1.8 \
+ data["pendiente_promedio"]*0.7 \
+ data["tipo_via"].map(via_factor) \
+ ruido

data["tiempo_real_min"] = \
data["tiempo_real_min"].clip(8,180)

# -----------------------------
# 9. SAVE MASTER DATASET
# -----------------------------

output_folder = r"K:\Dashboard Delivery\data\processed"

os.makedirs(output_folder,exist_ok=True)

output_path = os.path.join(
    output_folder,
    "dataset_maestro_intrak_bogota.csv"
)

data.to_csv(output_path,index=False)

print("\n✅ DATASET MAESTRO GENERADO")
print("📦 Guardado en:")
print(output_path)