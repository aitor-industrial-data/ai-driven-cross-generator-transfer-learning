# Plan de Proyecto: Predicción de Fallos en Turbinas Eólicas
## Kelmarsh SCADA — Guía paso a paso

**Objetivo**: modelo que dado el estado actual de los sensores, prediga si un fallo ocurrirá en las próximas X horas.  
**Filosofía**: simple primero, complejo después. El 80% del valor viene del 20% del trabajo.

---

## ARQUITECTURA FINAL (spoiler para entender las decisiones)

```
telemetry_filtered.parquet   ← 60 columnas × 9 años × 10 min
       +
fault_log.parquet            ← timestamp, turbine_id, fault_code
       ↓
  JOIN por fecha
       ↓
dataset_labeled.parquet      ← telemetría + ventanas rodantes + etiquetas
       ↓
  Un modelo por familia de fallos (5–7 modelos, no 40)
```

---

## FASE 0 — ENTORNO (1 tarde)

### 0.1 Instala esto y solo esto

```bash
pip install pandas pyarrow scikit-learn lightgbm matplotlib seaborn jupyter
```

Usa Jupyter Notebook. No hace falta nada más.

### 0.2 Estructura de carpetas

```
proyecto_turbinas/
├── data/
│   ├── raw/              ← csvs originales sin tocar NUNCA
│   │   ├── bronze/       ← tus carpetas Kelmarsh_SCADA_YYYY_XXXX
│   │   └── fault_log/    ← el CSV de fallos con is_failure_target
│   ├── processed/        ← lo que vas generando
│   └── models/           ← modelos guardados (.pkl)
├── notebooks/
│   ├── 01_merge.ipynb
│   ├── 02_features.ipynb
│   ├── 03_labeling.ipynb
│   ├── 04_train.ipynb
│   └── 05_evaluate.ipynb
└── src/
    └── utils.py          ← funciones reutilizables
```

**Regla de oro**: raw/ es sagrado. Nunca escribas en raw/.

---

## FASE 1 — UNIR TELEMETRÍA + FALLOS (1 día)

### 1.1 Carga y concatena todos los años de telemetría

```python
# notebook 01_merge.ipynb
import pandas as pd
import glob

# Carga todos los CSVs de telemetría
files = glob.glob("data/raw/bronze/**/Kelmarsh*.csv", recursive=True)
dfs = []
for f in files:
    df = pd.read_csv(f, parse_dates=["Date and time"], low_memory=False)
    # Extrae turbina del nombre de carpeta
    turbine_id = f.split("/")[-2]  # ajusta según tu ruta
    df["turbine_id"] = turbine_id
    dfs.append(df)

telem = pd.concat(dfs, ignore_index=True)
telem = telem.sort_values(["turbine_id", "Date and time"])
telem = telem.set_index(["turbine_id", "Date and time"])
```

### 1.2 Filtra las 60 columnas

```python
COLS_60 = [
    # VIENTO
    "Wind speed (m/s)", "Wind speed, Standard deviation (m/s)",
    "Wind speed Sensor 1 (m/s)", "Wind speed Sensor 1, Standard deviation (m/s)",
    "Wind speed Sensor 1, Minimum (m/s)", "Wind speed Sensor 1, Maximum (m/s)",
    "Wind speed Sensor 2 (m/s)", "Wind speed Sensor 2, Standard deviation (m/s)",
    "Wind speed Sensor 2, Minimum (m/s)", "Wind speed Sensor 2, Maximum (m/s)",
    "Wind direction (°)", "Wind direction, Standard deviation (°)",
    "Nacelle position (°)", "Nacelle position, Standard deviation (°)",
    "Vane position 1+2 (°)", "Vane position 1+2, StdDev (°)",
    # POTENCIA
    "Power (kW)", "Power, Standard deviation (kW)",
    "Power factor (cosphi)", "Power factor (cosphi), Standard deviation",
    "Reactive power (kvar)", "Grid voltage (V)", "Grid frequency (Hz)",
    "Current L1 / U (A)", "Current L2 / V (A)", "Current L3 / W (A)",
    # PITCH
    "Blade angle (pitch position) A (°)", "Blade angle (pitch position) A, Max (°)",
    "Blade angle (pitch position) A, Standard deviation (°)",
    "Blade angle (pitch position) B (°)", "Blade angle (pitch position) B, Standard deviation (°)",
    "Blade angle (pitch position) C (°)", "Blade angle (pitch position) C, Standard deviation (°)",
    "Motor current axis 1 (A)", "Motor current axis 1, Max (A)", "Motor current axis 1, StdDev (A)",
    "Motor current axis 2 (A)", "Motor current axis 2, Max (A)", "Motor current axis 2, StdDev (A)",
    "Motor current axis 3 (A)", "Motor current axis 3, Max (A)", "Motor current axis 3, StdDev (A)",
    "Temperature motor axis 1 (°C)", "Temperature motor axis 2 (°C)", "Temperature motor axis 3 (°C)",
    # TREN
    "Drive train acceleration (mm/ss)", "Tower Acceleration X (mm/ss)", "Tower Acceleration y (mm/ss)",
    "Generator RPM (RPM)", "Generator RPM, Standard deviation (RPM)",
    "Gear oil temperature (°C)", "Gear oil temperature, Max (°C)",
    "Gear oil inlet temperature (°C)", "Gear oil inlet pressure (bar)",
    "Gear oil inlet pressure, Min (bar)", "Gear oil pump pressure (bar)",
    "Metal particle count", "Front bearing temperature (°C)", "Rear bearing temperature (°C)",
    # GENERADOR Y CONVERTIDOR
    "Generator bearing front temperature (°C)", "Generator bearing front temperature, Max (°C)",
    "Generator bearing rear temperature (°C)", "Generator bearing rear temperature, Max (°C)",
    "Ambient temperature (converter) (°C)", "Ambient temperature (converter), Max (°C)",
    "Ambient temperature (converter), StdDev (°C)",
    # NACELLE Y TRANSFORMADOR
    "Nacelle temperature (°C)", "Nacelle temperature, Max (°C)",
    "Nacelle temperature, Standard deviation (°C)",
    "Nacelle ambient temperature (°C)",
    "Transformer temperature (°C)",
    # CABLE Y YAW
    "Cable windings from calibration point",
]

telem = telem[[c for c in COLS_60 if c in telem.columns]]
```

### 1.3 Guarda en Parquet (10x más rápido que CSV)

```python
telem.reset_index().to_parquet("data/processed/telemetry_filtered.parquet", index=False)
# ~3.7 GB CSV → ~400 MB Parquet. Se lee en segundos.
```

### 1.4 Carga el log de fallos

```python
faults = pd.read_csv("data/raw/fault_log/fault_log.csv", parse_dates=["timestamp"])
# Asegúrate de que tiene: timestamp, turbine_id, fault_code, is_failure_target
faults = faults[faults["is_failure_target"] == True]
faults = faults.sort_values(["turbine_id", "timestamp"])
faults.to_parquet("data/processed/faults_filtered.parquet", index=False)
```

---

## FASE 2 — ETIQUETADO (el corazón del proyecto, 2 días)

### Por qué NO hacer binario simple

Un binario `is_failure = True/False` te dice "algo pasó" pero el modelo no aprende
cuándo ni por qué. Peor: el 99.9% de las filas son False → el modelo aprende a decir
siempre False y tiene 99.9% de accuracy. Es una trampa.

### Lo que vas a crear en su lugar

Dos columnas por cada FAMILIA de fallos (no por cada código):

```
hours_to_fault_FAMILIA   ← cuántas horas faltan para el próximo fallo (NaN si no hay fallo próximo)
is_pre_fault_FAMILIA     ← True si hours_to_fault < lead_time de esa familia
```

### Las 6 familias de fallos (agrupa los 40 códigos en grupos)

```python
FAULT_FAMILIES = {
    "pitch":        {"codes": [675, 681, 682, 683, 697, 716, 717, 718],   "lead_hours": 240},  # 10 días
    "drivetrain":   {"codes": [59, 1070, 4500, 4510, 4520, 4530, 4540],    "lead_hours": 336},  # 14 días
    "converter":    {"codes": [785, 3110, 3205, 3220],                      "lead_hours": 72},   # 3 días
    "thermal":      {"codes": [1810, 2550, 2650, 2674, 3870],               "lead_hours": 120},  # 5 días
    "yaw_cable":    {"codes": [3160, 6052, 6054, 6120],                     "lead_hours": 168},  # 7 días
    "sensors":      {"codes": [6515, 6525, 6530, 6620, 6622, 6635],         "lead_hours": 72},   # 3 días
    "hydraulic":    {"codes": [2000, 2125, 5510, 5720],                     "lead_hours": 72},   # 3 días
}
```

### Código de etiquetado

```python
# notebook 03_labeling.ipynb
import pandas as pd
import numpy as np

telem = pd.read_parquet("data/processed/telemetry_filtered.parquet")
faults = pd.read_parquet("data/processed/faults_filtered.parquet")

telem["timestamp"] = pd.to_datetime(telem["Date and time"])
telem = telem.sort_values(["turbine_id", "timestamp"])

def label_family(telem_turb, faults_turb, lead_hours):
    """Para una turbina y una familia, calcula hours_to_fault en cada fila."""
    fault_times = faults_turb["timestamp"].sort_values().values
    timestamps = telem_turb["timestamp"].values
    
    hours_to_fault = np.full(len(timestamps), np.nan)
    
    for i, ts in enumerate(timestamps):
        # Próximo fallo después de este timestamp
        future = fault_times[fault_times > ts]
        if len(future) == 0:
            continue
        next_fault = future[0]
        delta_hours = (next_fault - ts) / np.timedelta64(1, 'h')
        if delta_hours <= lead_hours:
            hours_to_fault[i] = delta_hours
    
    return hours_to_fault

# Aplica para cada turbina y cada familia
results = []
for turbine in telem["turbine_id"].unique():
    t = telem[telem["turbine_id"] == turbine].copy()
    f = faults[faults["turbine_id"] == turbine].copy()
    
    for family, cfg in FAULT_FAMILIES.items():
        f_family = f[f["fault_code"].isin(cfg["codes"])]
        hours = label_family(t, f_family, cfg["lead_hours"])
        t[f"hours_to_{family}"] = hours
        t[f"is_pre_{family}"] = hours <= cfg["lead_hours"]
    
    results.append(t)

labeled = pd.concat(results, ignore_index=True)
labeled.to_parquet("data/processed/dataset_labeled.parquet", index=False)
print(f"Filas totales: {len(labeled):,}")
print(labeled[[c for c in labeled.columns if "is_pre_" in c]].sum())
```

---

## FASE 3 — FEATURES DE VENTANA RODANTE (1–2 días)

### Tu idea era buena, solo hay que hacerla por familia

No necesitas las 12 columnas para los 60 sensores para todos los modelos.
Cada familia usa un subconjunto de sensores. Esto reduce el problema de ~720 features
a ~60–80 por modelo.

```python
# Define qué sensores importan para cada familia
FAMILY_SENSORS = {
    "pitch": [
        "Blade angle (pitch position) A (°)", "Blade angle (pitch position) A, Standard deviation (°)",
        "Blade angle (pitch position) B (°)", "Blade angle (pitch position) B, Standard deviation (°)",
        "Blade angle (pitch position) C (°)", "Blade angle (pitch position) C, Standard deviation (°)",
        "Motor current axis 1 (A)", "Motor current axis 1, Max (A)",
        "Motor current axis 2 (A)", "Motor current axis 2, Max (A)",
        "Motor current axis 3 (A)", "Motor current axis 3, Max (A)",
        "Temperature motor axis 1 (°C)", "Temperature motor axis 2 (°C)", "Temperature motor axis 3 (°C)",
        "Power (kW)", "Wind speed (m/s)",
    ],
    "drivetrain": [
        "Drive train acceleration (mm/ss)", "Tower Acceleration X (mm/ss)", "Tower Acceleration y (mm/ss)",
        "Gear oil temperature (°C)", "Gear oil temperature, Max (°C)",
        "Gear oil inlet temperature (°C)", "Gear oil inlet pressure (bar)",
        "Metal particle count", "Front bearing temperature (°C)", "Rear bearing temperature (°C)",
        "Generator RPM (RPM)", "Generator RPM, Standard deviation (RPM)",
        "Power (kW)", "Wind speed (m/s)",
    ],
    "converter": [
        "Ambient temperature (converter) (°C)", "Ambient temperature (converter), Max (°C)",
        "Ambient temperature (converter), StdDev (°C)",
        "Power factor (cosphi)", "Power factor (cosphi), Standard deviation",
        "Reactive power (kvar)", "Current L1 / U (A)", "Current L2 / V (A)", "Current L3 / W (A)",
        "Power (kW)", "Wind speed (m/s)",
    ],
    "thermal": [
        "Generator bearing front temperature (°C)", "Generator bearing front temperature, Max (°C)",
        "Generator bearing rear temperature (°C)", "Generator bearing rear temperature, Max (°C)",
        "Nacelle temperature (°C)", "Nacelle temperature, Max (°C)",
        "Nacelle ambient temperature (°C)", "Transformer temperature (°C)",
        "Gear oil temperature (°C)", "Gear oil inlet temperature (°C)",
        "Power (kW)", "Wind speed (m/s)",
    ],
    "yaw_cable": [
        "Nacelle position (°)", "Nacelle position, Standard deviation (°)",
        "Wind direction (°)", "Wind direction, Standard deviation (°)",
        "Vane position 1+2 (°)", "Vane position 1+2, StdDev (°)",
        "Cable windings from calibration point",
        "Power (kW)", "Wind speed (m/s)",
    ],
    "sensors": [
        "Wind speed Sensor 1 (m/s)", "Wind speed Sensor 1, Standard deviation (m/s)",
        "Wind speed Sensor 1, Minimum (m/s)", "Wind speed Sensor 1, Maximum (m/s)",
        "Wind speed Sensor 2 (m/s)", "Wind speed Sensor 2, Standard deviation (m/s)",
        "Vane position 1+2 (°)", "Vane position 1+2, StdDev (°)",
        "Vane position 1+2, Max (°)", "Vane position 1+2, Min (°)",
        "Wind direction (°)", "Wind direction, Standard deviation (°)",
        "Power (kW)", "Wind speed (m/s)",
    ],
    "hydraulic": [
        "Gear oil inlet pressure (bar)", "Gear oil inlet pressure, Min (bar)",
        "Gear oil pump pressure (bar)", "Gear oil inlet temperature (°C)",
        "Generator RPM (RPM)", "Generator RPM, Standard deviation (RPM)",
        "Power (kW)",
    ],
}

# Ventanas en número de pasos de 10 min
WINDOWS = {
    "1h":   6,
    "5h":   30,
    "24h":  144,
    "7d":   1008,
}

def rolling_features(df, sensors, windows):
    """Genera media, std y pendiente para cada sensor en cada ventana."""
    new_cols = {}
    for sensor in sensors:
        if sensor not in df.columns:
            continue
        s = df[sensor]
        for name, w in windows.items():
            new_cols[f"{sensor}__mean_{name}"]  = s.rolling(w, min_periods=w//2).mean()
            new_cols[f"{sensor}__std_{name}"]   = s.rolling(w, min_periods=w//2).std()
            # Pendiente: regresión lineal simplificada = (último - primero) / ventana
            new_cols[f"{sensor}__slope_{name}"] = (
                s.rolling(w, min_periods=w//2).apply(
                    lambda x: (x.iloc[-1] - x.iloc[0]) / len(x) if len(x) > 1 else np.nan,
                    raw=False
                )
            )
    return pd.DataFrame(new_cols, index=df.index)

# Genera features por familia (hazlo turbina a turbina para no quedarte sin RAM)
labeled = pd.read_parquet("data/processed/dataset_labeled.parquet")
labeled = labeled.sort_values(["turbine_id", "timestamp"])

for family, sensors in FAMILY_SENSORS.items():
    print(f"Generando features para familia: {family}")
    feat_list = []
    for turbine in labeled["turbine_id"].unique():
        t = labeled[labeled["turbine_id"] == turbine]
        feat = rolling_features(t, sensors, WINDOWS)
        feat_list.append(feat)
    all_feats = pd.concat(feat_list)
    # Guarda por familia (no junto todo, o te quedas sin RAM)
    all_feats.to_parquet(f"data/processed/features_{family}.parquet")
    print(f"  → {all_feats.shape[1]} features generadas")
```

### Cuántas features por modelo resultantes

| Familia | Sensores base | × 4 ventanas × 3 stats | Total features |
|---|---|---|---|
| pitch | 17 | × 12 | ~204 |
| drivetrain | 14 | × 12 | ~168 |
| converter | 11 | × 12 | ~132 |
| thermal | 12 | × 12 | ~144 |
| yaw_cable | 9 | × 12 | ~108 |
| sensors | 14 | × 12 | ~168 |
| hydraulic | 7 | × 12 | ~84 |

Manejable. LightGBM lo entrena en minutos en un portátil.

---

## FASE 4 — ENTRENAMIENTO (1 día)

### 4.1 El target correcto

Usa `is_pre_FAMILIA` como target binario **pero con pesos de clase**,
no con undersampling. Esto resuelve el desequilibrio sin tirar datos.

```python
from sklearn.utils.class_weight import compute_class_weight

y = df["is_pre_pitch"]
weights = compute_class_weight("balanced", classes=[False, True], y=y)
# Resultado típico: True pesa ~50x más que False. LightGBM lo maneja bien.
```

### 4.2 Split temporal (NUNCA random)

```python
# Corte temporal: 80% para train, 20% para test
# Siempre el futuro en test, nunca mezcles fechas
cutoff = df["timestamp"].quantile(0.8)
train = df[df["timestamp"] < cutoff]
test  = df[df["timestamp"] >= cutoff]
```

### 4.3 Entrena con LightGBM

```python
import lightgbm as lgb
from sklearn.metrics import classification_report

family = "drivetrain"  # el más interesante para empezar

# Carga datos de esa familia
base = labeled[["turbine_id", "timestamp", f"is_pre_{family}"]].copy()
feats = pd.read_parquet(f"data/processed/features_{family}.parquet")
df = base.join(feats).dropna(subset=[f"is_pre_{family}"])

cutoff = df["timestamp"].quantile(0.8)
X_train = df[df["timestamp"] < cutoff][feats.columns]
y_train = df[df["timestamp"] < cutoff][f"is_pre_{family}"]
X_test  = df[df["timestamp"] >= cutoff][feats.columns]
y_test  = df[df["timestamp"] >= cutoff][f"is_pre_{family}"]

model = lgb.LGBMClassifier(
    n_estimators=500,
    class_weight="balanced",
    learning_rate=0.05,
    num_leaves=63,
    random_state=42,
    n_jobs=-1,
)
model.fit(X_train, y_train, eval_set=[(X_test, y_test)])

print(classification_report(y_test, model.predict(X_test)))
import pickle
pickle.dump(model, open(f"data/models/model_{family}.pkl", "wb"))
```

### 4.4 Métrica correcta para este problema

**No uses accuracy.** Usa:
- **Recall** (de los fallos reales, ¿cuántos detectaste?) → quieres >80%
- **Precision** (de las alertas que lanzaste, ¿cuántas eran reales?) → quieres >60%
- **F1** como balance

Un Recall de 85% con Precision de 65% es un modelo excelente para este dominio.

---

## FASE 5 — EVALUAR E INTERPRETAR (1 día)

### 5.1 Feature importance (qué sensores importan de verdad)

```python
import matplotlib.pyplot as plt

feat_imp = pd.Series(model.feature_importances_, index=feats.columns)
feat_imp.nlargest(20).plot(kind="barh", figsize=(10, 8))
plt.title("Top 20 features — Drivetrain family")
plt.tight_layout()
plt.savefig("data/models/feature_importance_drivetrain.png")
```

Esto te dirá si el modelo usa las señales que tiene sentido físico que use.
Si los top features son `Metal particle count` y `Gear oil temperature` → el modelo es fiable.
Si el top feature es `Grid frequency` → algo está mal en el etiquetado.

### 5.2 Curva de alertas en el tiempo

```python
# Para una turbina de test, pinta la probabilidad predicha vs el fallo real
turb_test = df[(df["turbine_id"] == "Kelmarsh_SCADA_2022_4457") & 
               (df["timestamp"] >= cutoff)]
proba = model.predict_proba(turb_test[feats.columns])[:, 1]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 6), sharex=True)
ax1.plot(turb_test["timestamp"], proba, label="Probabilidad de fallo")
ax1.axhline(0.5, color="red", linestyle="--", label="Umbral 50%")
ax1.set_ylabel("P(fallo próximo)")
ax1.legend()

fault_times = faults[(faults["turbine_id"] == "Kelmarsh_SCADA_2022_4457") & 
                      (faults["fault_code"].isin(FAULT_FAMILIES["drivetrain"]["codes"]))]
for ft in fault_times["timestamp"]:
    ax1.axvline(ft, color="red", alpha=0.5)

plt.tight_layout()
plt.savefig("data/models/alert_timeline.png")
```

Esto es lo que le enseñas a un operador de planta. Si la curva sube 2 semanas antes
del palo rojo, el modelo funciona.

---

## RESUMEN DE TIEMPOS Y DIFICULTAD

| Fase | Tiempo estimado | Dificultad | Puede fallar si... |
|---|---|---|---|
| 0 — Entorno | 2 horas | Baja | Versión Python incompatible |
| 1 — Merge | 1 día | Media | turbine_id no coincide entre archivos |
| 2 — Etiquetado | 2 días | Alta | Timestamps desalineados entre telemetría y fallos |
| 3 — Features | 1–2 días | Media | RAM insuficiente (hazlo turbina a turbina) |
| 4 — Entrenamiento | 1 día | Baja | Target muy desbalanceado (usa class_weight) |
| 5 — Evaluación | 1 día | Media | Confundir accuracy con recall |

**Total: ~1–2 semanas a tiempo parcial desde casa.**

---

## RIESGOS Y CÓMO EVITARLOS

### Riesgo 1: Te quedas sin RAM
- Procesa siempre turbina a turbina
- Usa Parquet, no CSV
- Nunca cargues el dataset entero en memoria

### Riesgo 2: El modelo siempre predice False
- Verifica que `is_pre_FAMILIA.value_counts()` tenga >100 casos True
- Usa `class_weight="balanced"` siempre
- Si hay <50 fallos de un tipo, no entrenes ese modelo

### Riesgo 3: El modelo hace "trampa" (data leakage)
- NUNCA uses random split — siempre temporal
- No incluyas columnas derivadas del target como feature
- Las ventanas rodantes se calculan solo sobre el pasado (el `rolling()` por defecto es hacia atrás)

### Riesgo 4: Timestamps desalineados
- Verifica que ambos datasets usen UTC o ambos usen local
- Redondea timestamps a 10 min antes del join: `df["timestamp"].dt.round("10min")`

### Riesgo 5: El fallo que tiene 10 ocurrencias en 9 años
- Empieza por drivetrain (1070) y pitch (675) — son los más frecuentes
- Los fallos raros (2000, 3160) déjalos para después o ignóralos

