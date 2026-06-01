# Plan Definitivo — Predicción de Fallos Turbina Kelmarsh T1
## Dataset: 2018–2021 · 210.388 filas · 10 min · 1 turbina

---

### FASE 0 — ENTORNO (2 horas)
 
**0.1 Instala exactamente esto:**
```bash
pip install pandas pyarrow scikit-learn lightgbm matplotlib seaborn jupyter
```
 
**0.2 Estructura de carpetas:**
```
proyecto_turbinas/
├── data/
│   ├── raw/                      ← NUNCA tocar
│   │   ├── bronze/               ← CSVs telemetría por año
│   │   ├── silver/
│   │   │   ├── fault_log.csv     ← archivo con is_failure_target
│   │   │   └── technical_fault_catalog.csv
│   ├── processed/                ← lo que va generando
│   └── models/                   ← modelos .pkl
├── notebooks/
│   ├── 01_eda_status_and_events.ipynb                # Estados operativos, status, eventos temporales
│   ├── 02_02_eda_telemetry_and_sensors.ipynb         # Telemetría, sensores, análisis de nulos
│   ├── 03_merge_and_cleaning.ipynb                   # Merge de fuentes + limpieza
│   ├── 04_labeling.ipynb                             # Etiquetado de fallos/estados objetivo
│   ├── 05_feature_engineering.ipynb                  # Ingeniería de características
│   ├── 06_train_yaw.ipynb                            # Modelo: desalineación de yaw
│   ├── 07_train_generator.ipynb                      # Modelo: fallos de generador
│   ├── 08_train_brake.ipynb                          # Modelo: fallos de freno
│   └── 09_train_pitch.ipynb                          # Modelo: fallos de pitch
```
 
---
 
### FASE 1 — MERGE TELEMETRÍA + FALLOS (notebook 03)
 
**Paso 1.1 — Carga y concatena todos los CSVs de telemetría**
```python
import pandas as pd
import glob
 
# Lee todos los CSVs de la carpeta bronze
files = sorted(glob.glob("data/raw/bronze/**/*.csv", recursive=True))
print(f"Archivos encontrados: {len(files)}")
 
chunks = []
for f in files:
    df = pd.read_csv(f, parse_dates=["Date and time"], low_memory=False)
    chunks.append(df)
 
telem = pd.concat(chunks, ignore_index=True)
telem = telem.sort_values("Date and time").reset_index(drop=True)
telem = telem.rename(columns={"Date and time": "timestamp"})
 
print(f"Filas totales: {len(telem):,}")
print(f"Rango: {telem['timestamp'].min()} → {telem['timestamp'].max()}")
```
 
**Paso 1.2 — Renombra columnas a snake_case (las 303 columnas originales tienen espacios)**
 
Las columnas del CSV original tienen nombres con paréntesis, espacios y caracteres
especiales. Ya tienes el análisis de nulos con nombres en snake_case — el dataset
procesado con Spark ya los renombró. Verifica cuál formato tienes:
 
```python
# Comprueba si las columnas ya están en snake_case o siguen el original
print(telem.columns[:5].tolist())
# Si ves "Date and time" → original con espacios
# Si ves "Date_and_time" → ya procesado
```
 
Si siguen con espacios, renombra:
```python
telem.columns = (telem.columns
    .str.replace(r'[^\w]', '_', regex=True)
    .str.replace(r'_+', '_', regex=True)
    .str.strip('_'))
```
 
**Paso 1.3 — Filtra las columnas que necesitas**
 
```python
# Columnas base para TODOS los modelos — <10% nulos, disponibles siempre
COLS_BASE = [
    "timestamp",
    # Viento
    "Wind_speed_ms", "Wind_speed_Standard_deviation_ms",
    "Wind_speed_Sensor_1_ms", "Wind_speed_Sensor_2_ms",
    "Wind_direction", "Wind_direction_Standard_deviation",
    "Nacelle_position", "Nacelle_position_Standard_deviation",
    "Vane_position_12",
    # Potencia
    "Power_kW", "Power_Standard_deviation_kW",
    "Power_factor_cosphi", "Reactive_power_kvar",
    "Grid_voltage_V", "Grid_frequency_Hz",
    # Generador y tren
    "Generator_RPM_RPM", "Generator_RPM_Standard_deviation_RPM",
    "Rotor_speed_RPM",
    "Drive_train_acceleration_mmss",
    # Temperaturas — todas <5% nulos
    "Generator_bearing_front_temperature_C", "Generator_bearing_rear_temperature_C",
    "Generator_bearing_front_temperature_Max_C", "Generator_bearing_rear_temperature_Max_C",
    "Nacelle_temperature_C", "Nacelle_temperature_Max_C",
    "Nacelle_ambient_temperature_C",
    "Ambient_temperature_converter_C",
    "Front_bearing_temperature_C", "Rear_bearing_temperature_C",
    "Gear_oil_temperature_C",
    "Gear_oil_inlet_temperature_C",
    "Stator_temperature_1_C",
    "Temp_top_box_C",
    # Hidráulico
    "Gear_oil_inlet_pressure_bar", "Gear_oil_pump_pressure_bar",
    # Cable y pitch
    "Cable_windings_from_calibration_point",
    "Blade_angle_pitch_position_A", "Blade_angle_pitch_position_B", "Blade_angle_pitch_position_C",
    "Motor_current_axis_1_A", "Motor_current_axis_2_A", "Motor_current_axis_3_A",
    "Temperature_motor_axis_1_C", "Temperature_motor_axis_2_C",
    # Partículas metálicas — 0% nulos
    "Metal_particle_count",
]
 
telem = telem[[c for c in COLS_BASE if c in telem.columns]].copy()
print(f"Columnas seleccionadas: {telem.shape[1]}")
print(f"Nulos por columna:\n{telem.isnull().mean().sort_values(ascending=False).head(10)}")
```
 
**Paso 1.4 — Guarda en Parquet**
```python
telem.to_parquet("data/processed/telemetry_clean.parquet", index=False)
# De ~3.7 GB CSV → ~200 MB Parquet. Se carga en <5 segundos.
```
 
**Paso 1.5 — Carga y prepara el fault_log**
```python
faults = pd.read_csv("data/raw/silver/fault_log.csv",
                     parse_dates=["Timestamp start"])
 
# Redondear a la baja a 10 minutos (no round — floor)
faults["timestamp"] = faults["Timestamp start"].dt.floor("10min")
 
# Filtrar solo los que son failure target
targets = faults[faults["is_failure_target"] == True].copy()
targets = targets[["timestamp", "Code", "Message", "Status"]].copy()
 
print(f"Eventos target totales: {len(targets)}")
print(targets["Code"].value_counts().head(10))
 
targets.to_parquet("data/processed/fault_targets.parquet", index=False)
```
 
**VERIFICACIÓN CRÍTICA — hacer siempre:**
```python
# Comprueba que los timestamps encajan
telem_times = set(telem["timestamp"].dt.floor("10min").unique())
fault_times = set(targets["timestamp"].unique())
overlap = fault_times.intersection(telem_times)
print(f"Eventos target que tienen telemetría: {len(overlap)}/{len(fault_times)}")
# Si es 0 → hay problema de timezone o formato de fecha. Investigar antes de continuar.
```
 
---
 
### FASE 2 — ETIQUETADO (notebook 04)
 
**Por qué NO usar binario simple y qué usar en cambio:**
 
El binario (0/1) tiene dos problemas: el 99% de las filas son 0 y el modelo aprende
a decir siempre 0, y no puedes ajustar el umbral de alerta después sin reentrenar.
 
Usamos `hours_to_fault`: cuántas horas faltan al próximo fallo de esa familia.
- Si no hay fallo en las próximas N horas → NaN (estado normal)
- Si hay fallo en las próximas N horas → el número de horas (ej: 47.3)
- Luego `is_pre_fault = hours_to_fault <= lead_time` → esto sí es binario, pero calculado
**Paso 2.1 — Define familias y lead times**
```python
FAULT_FAMILIES = {
    "yaw_cable":   {"codes": [6052, 6200, 6054, 6120, 6300],           "lead_hours": 168},
    "brake_hydro": {"codes": [2125, 5720, 5510, 2000, 1860],           "lead_hours": 120},
    "generator":   {"codes": [3000, 2550, 2650, 2655, 2674, 8400, 3125], "lead_hours": 120},
    "pitch_bat":   {"codes": [716, 717, 718, 681, 682, 683, 675, 785, 850], "lead_hours": 336},
}
```
 
**Paso 2.2 — Función de etiquetado**
```python
import numpy as np
 
telem = pd.read_parquet("data/processed/telemetry_clean.parquet")
targets = pd.read_parquet("data/processed/fault_targets.parquet")
 
telem = telem.sort_values("timestamp").reset_index(drop=True)
 
def label_family(telem_df, fault_times_sorted, lead_hours):
    """
    Para cada fila de telemetría, calcula horas al próximo fallo.
    fault_times_sorted: array numpy de timestamps ordenados
    """
    ts_array = telem_df["timestamp"].values.astype("datetime64[ns]")
    fault_array = np.array(fault_times_sorted, dtype="datetime64[ns]")
    lead_ns = np.timedelta64(int(lead_hours * 3600e9), "ns")
    
    hours_arr = np.full(len(ts_array), np.nan)
    
    for i, ts in enumerate(ts_array):
        # Busca el primer fallo después de este timestamp
        future_mask = fault_array > ts
        if not future_mask.any():
            continue
        next_fault = fault_array[future_mask][0]
        delta_h = (next_fault - ts) / np.timedelta64(1, "h")
        if delta_h <= lead_hours:
            hours_arr[i] = delta_h
    
    return hours_arr
 
# Aplica para cada familia
for family, cfg in FAULT_FAMILIES.items():
    print(f"Etiquetando familia: {family}...")
    
    fault_times = (targets[targets["Code"].isin(cfg["codes"])]
                   ["timestamp"]
                   .sort_values()
                   .values)
    
    hours = label_family(telem, fault_times, cfg["lead_hours"])
    telem[f"hours_to_{family}"] = hours
    telem[f"is_pre_{family}"] = (hours <= cfg["lead_hours"]) & (~np.isnan(hours))
    
    n_true = telem[f"is_pre_{family}"].sum()
    n_total = len(telem)
    print(f"  → {n_true:,} filas pre-fallo ({100*n_true/n_total:.2f}% del total)")
 
telem.to_parquet("data/processed/dataset_labeled.parquet", index=False)
print("Guardado.")
```
 
**Paso 2.3 — Verifica el etiquetado visualmente para 1 familia**
```python
import matplotlib.pyplot as plt
 
# Comprueba visualmente que el etiquetado tiene sentido
df = pd.read_parquet("data/processed/dataset_labeled.parquet")
 
# Muestra un fallo concreto y sus horas previas
family = "yaw_cable"
pre = df[df[f"is_pre_{family}"] == True]
print(f"Primeras filas etiquetadas como pre-{family}:")
print(pre[["timestamp", f"hours_to_{family}", "Nacelle_position",
           "Cable_windings_from_calibration_point"]].head(20))
 
# Pinta Cable_windings en los 7 días antes de un fallo
fault_sample = targets[targets["Code"] == 6200]["timestamp"].iloc[0]
window = df[(df["timestamp"] >= fault_sample - pd.Timedelta(days=10)) &
            (df["timestamp"] <= fault_sample + pd.Timedelta(hours=1))]
 
plt.figure(figsize=(14, 4))
plt.plot(window["timestamp"], window["Cable_windings_from_calibration_point"])
plt.axvline(fault_sample, color="red", label="Fallo 6200")
plt.title("Cable windings antes de un Cable autounwind")
plt.legend()
plt.tight_layout()
plt.savefig("data/processed/check_etiquetado_cable.png")
plt.show()
```
 
---
 
### FASE 3 — FEATURES DE VENTANA RODANTE (notebook 03)
 
**Paso 3.1 — Función de rolling features**
 
```python
import pandas as pd
import numpy as np
 
df = pd.read_parquet("data/processed/dataset_labeled.parquet")
df = df.sort_values("timestamp").reset_index(drop=True)
 
# Ventanas en número de pasos de 10 min
WINDOWS = {"1h": 6, "6h": 36, "24h": 144, "7d": 1008}
 
def make_rolling_features(df, sensors, windows):
    """
    Genera media, std y pendiente para cada sensor en cada ventana.
    IMPORTANTE: rolling() en pandas mira hacia atrás por defecto — correcto.
    """
    feats = {}
    for col in sensors:
        if col not in df.columns:
            print(f"  WARN: {col} no encontrada, saltando")
            continue
        s = df[col]
        for wname, w in windows.items():
            roll = s.rolling(w, min_periods=max(1, w//3))
            feats[f"{col}__mean_{wname}"]  = roll.mean()
            feats[f"{col}__std_{wname}"]   = roll.std()
            # Pendiente simplificada: (último - primero) / n_pasos
            feats[f"{col}__slope_{wname}"] = roll.apply(
                lambda x: (x.iloc[-1] - x.iloc[0]) / len(x) if len(x) > 1 else 0.0,
                raw=False
            )
    return pd.DataFrame(feats, index=df.index)
```
 
**Paso 3.2 — Define sensores por familia y genera features**
 
```python
FAMILY_SENSORS = {
    "yaw_cable": [
        "Nacelle_position", "Nacelle_position_Standard_deviation",
        "Wind_direction", "Wind_direction_Standard_deviation",
        "Vane_position_12", "Cable_windings_from_calibration_point",
        "Wind_speed_ms", "Power_kW",
    ],
    "brake_hydro": [
        "Gear_oil_inlet_pressure_bar", "Gear_oil_pump_pressure_bar",
        "Gear_oil_inlet_temperature_C", "Gear_oil_temperature_C",
        "Generator_RPM_RPM", "Generator_RPM_Standard_deviation_RPM",
        "Rotor_speed_RPM", "Power_kW",
        "Front_bearing_temperature_C", "Rear_bearing_temperature_C",
        "Metal_particle_count",
    ],
    "generator": [
        "Generator_bearing_front_temperature_C", "Generator_bearing_rear_temperature_C",
        "Generator_bearing_front_temperature_Max_C", "Generator_bearing_rear_temperature_Max_C",
        "Nacelle_temperature_C", "Nacelle_ambient_temperature_C",
        "Ambient_temperature_converter_C",
        "Power_kW", "Reactive_power_kvar", "Power_factor_cosphi",
        "Stator_temperature_1_C", "Wind_speed_ms",
    ],
    "pitch_bat": [
        "Motor_current_axis_1_A", "Motor_current_axis_2_A", "Motor_current_axis_3_A",
        "Blade_angle_pitch_position_A", "Blade_angle_pitch_position_B", "Blade_angle_pitch_position_C",
        "Temperature_motor_axis_1_C", "Temperature_motor_axis_2_C",
        "Nacelle_ambient_temperature_C", "Power_kW", "Wind_speed_ms",
    ],
    "tower": [
            "Drive_train_acceleration_mmss",
        "Generator_RPM_RPM", "Rotor_speed_RPM",
        "Wind_speed_ms", "Wind_speed_Standard_deviation_ms",
    ],
}
 
# AÑADE features calculadas (no están en el CSV, las construyes tú)
df["yaw_error"] = (df["Nacelle_position"] - df["Wind_direction"]).abs() % 360
df["yaw_error"] = df["yaw_error"].apply(lambda x: x if x <= 180 else 360 - x)
df["T_bearing_delta"] = df["Generator_bearing_front_temperature_C"] - df["Nacelle_ambient_temperature_C"]
df["pitch_asymmetry"] = (df[["Blade_angle_pitch_position_A",
                              "Blade_angle_pitch_position_B",
                              "Blade_angle_pitch_position_C"]].max(axis=1) -
                          df[["Blade_angle_pitch_position_A",
                              "Blade_angle_pitch_position_B",
                              "Blade_angle_pitch_position_C"]].min(axis=1))
 
# Añade las calculadas a los sensores de cada familia
FAMILY_SENSORS["yaw_cable"].append("yaw_error")
FAMILY_SENSORS["generator"].append("T_bearing_delta")
FAMILY_SENSORS["pitch_bat"].append("pitch_asymmetry")
 
# Genera y guarda features por familia
for family, sensors in FAMILY_SENSORS.items():
    print(f"\nGenerando features para: {family}")
    feats = make_rolling_features(df, sensors, WINDOWS)
    
    # Añade las columnas target de esta familia
    target_cols = [f"hours_to_{family}", f"is_pre_{family}"]
    output = pd.concat([df[["timestamp"] + target_cols], feats], axis=1)
    
    path = f"data/processed/features_{family}.parquet"
    output.to_parquet(path, index=False)
    print(f"  → {feats.shape[1]} features | guardado en {path}")
    print(f"  → Filas pre-fallo: {output[f'is_pre_{family}'].sum():,}")
```
 
---
 
### FASE 4 — ENTRENAMIENTO (un notebook por familia)
 
**Paso 4.1 — Carga datos y hace split temporal**
 
```python
import lightgbm as lgb
from sklearn.metrics import classification_report, confusion_matrix
import pickle
 
family = "yaw_cable"   # cambia esto por cada familia
 
df = pd.read_parquet(f"data/processed/features_{family}.parquet")
df = df.dropna(subset=[f"is_pre_{family}"])
 
# Features: todas las columnas que no sean timestamp ni target
feature_cols = [c for c in df.columns
                if c not in ["timestamp", f"hours_to_{family}", f"is_pre_{family}"]]
 
# Split temporal — NUNCA aleatorio
# Usa el 80% más antiguo para train, el 20% más reciente para test
cutoff = df["timestamp"].quantile(0.80)
print(f"Corte temporal: {cutoff}")
print(f"  Train: hasta {cutoff}")
print(f"  Test:  desde {cutoff}")
 
train = df[df["timestamp"] < cutoff]
test  = df[df["timestamp"] >= cutoff]
 
X_train = train[feature_cols]
y_train = train[f"is_pre_{family}"]
X_test  = test[feature_cols]
y_test  = test[f"is_pre_{family}"]
 
print(f"\nTrain: {len(train):,} filas | {y_train.sum():,} positivos ({100*y_train.mean():.2f}%)")
print(f"Test:  {len(test):,} filas  | {y_test.sum():,} positivos ({100*y_test.mean():.2f}%)")
```
 
**Paso 4.2 — Entrena LightGBM con class_weight**
 
```python
model = lgb.LGBMClassifier(
    n_estimators=500,
    class_weight="balanced",     # resuelve el desequilibrio sin tirar datos
    learning_rate=0.05,
    num_leaves=31,               # empieza conservador, sube si underfitting
    min_child_samples=20,        # evita overfitting con pocos positivos
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    verbose=-1,
)
 
model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    callbacks=[lgb.early_stopping(50, verbose=True), lgb.log_evaluation(50)],
)
 
# Guarda el modelo
pickle.dump(model, open(f"data/models/model_{family}.pkl", "wb"))
print(f"Modelo guardado: data/models/model_{family}.pkl")
```
 
**Paso 4.3 — Evalúa con las métricas correctas**
 
```python
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]
 
print("\n" + "="*50)
print(f"RESULTADOS — {family}")
print("="*50)
print(classification_report(y_test, y_pred, target_names=["Normal", "Pre-fallo"]))
 
cm = confusion_matrix(y_test, y_pred)
print(f"\nMatriz de confusión:")
print(f"  Verdaderos negativos: {cm[0,0]:,}  (alarmas correctas de 'todo OK')")
print(f"  Falsos positivos:     {cm[0,1]:,}  (falsas alarmas)")
print(f"  Falsos negativos:     {cm[1,0]:,}  (fallos NO detectados) ← el peor error")
print(f"  Verdaderos positivos: {cm[1,1]:,}  (fallos detectados correctamente)")
 
# Métricas objetivo:
# Recall > 0.80 (detectar >80% de los fallos reales)
# Precision > 0.50 (>50% de las alarmas son reales — aceptable en industria)
```
 
**Paso 4.4 — Feature importance (validación de sentido físico)**
 
```python
import matplotlib.pyplot as plt
 
feat_imp = pd.Series(model.feature_importances_, index=feature_cols)
top20 = feat_imp.nlargest(20)
 
fig, ax = plt.subplots(figsize=(10, 8))
top20.sort_values().plot(kind="barh", ax=ax, color="#378ADD")
ax.set_title(f"Top 20 features — {family}")
ax.set_xlabel("Importancia (gain)")
plt.tight_layout()
plt.savefig(f"data/models/feature_importance_{family}.png", dpi=150)
plt.show()
 
# VERIFICA: ¿las features más importantes tienen sentido físico?
# Yaw: deberían aparecer cable_windings, yaw_error, nacelle_std
# Si aparece Power_kW como top 1 → sospechoso, revisar etiquetado
```
 
**Paso 4.5 — Visualiza alertas en el tiempo**
 
```python
# Para el período de test, pinta la probabilidad predicha vs los fallos reales
test_with_prob = test.copy()
test_with_prob["prob"] = y_prob
 
fig, ax = plt.subplots(figsize=(16, 5))
ax.fill_between(test_with_prob["timestamp"], test_with_prob["prob"],
                alpha=0.4, color="#378ADD", label="P(fallo próximo)")
ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, label="Umbral 50%")
 
# Marca los fallos reales
fault_times_test = (targets[targets["Code"].isin(FAULT_FAMILIES[family]["codes"])]
                    [targets["timestamp"] >= cutoff]["timestamp"])
for ft in fault_times_test:
    ax.axvline(ft, color="red", alpha=0.7, linewidth=1.5)
 
ax.set_ylabel("Probabilidad de fallo")
ax.set_title(f"Alertas predichas vs fallos reales — {family} (período de test)")
ax.legend()
plt.tight_layout()
plt.savefig(f"data/models/timeline_{family}.png", dpi=150)
plt.show()
# Si la probabilidad sube ANTES de las líneas rojas → el modelo funciona
```
 
---
 
### RIESGOS Y CÓMO EVITARLOS
 
| Riesgo | Síntoma | Solución |
|--------|---------|---------|
| RAM insuficiente | Python se cuelga en Fase 3 | Procesar en chunks de 6 meses |
| Timestamps no encajan | overlap=0 en verificación Fase 1 | Comprobar timezone (UTC vs local) |
| Modelo siempre predice False | Accuracy 99%, Recall 0% | Confirmar `class_weight="balanced"` |
| Data leakage | Métricas perfectas en test | Verificar que el split sea temporal |
| Features del futuro | Rolling std sube justo EN el fallo | `rolling()` por defecto es lookback — correcto |
| Sensores con 53% nulos en rolling | NaN se propagan | NO incluir Max/StdDev con >40% nulos |
| Familia sensores (solo dic 2021) | Overfitting a un evento | Empezar con reglas deterministas |
 
### ORDEN FINAL RECOMENDADO
 
```
Semana 1: Fase 0 + Fase 1 (entorno + merge + parquet)
Semana 2: Fase 2 (etiquetado + verificación visual)
Semana 3: Fase 3 + 4 para YAW/CABLE (el más sencillo y más datos)
Semana 4: Fase 4 para GENERADOR y FRENO
Semana 5: Fase 4 para PITCH
Semana 6: Evaluación cruzada + ajuste de umbrales
```
