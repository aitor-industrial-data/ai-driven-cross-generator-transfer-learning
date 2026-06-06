# Guía de Despliegue en Producción — Transfer Learning a Turbina 2
## Sistema de Mantenimiento Predictivo · Kelmarsh Wind Farm

---

## Visión General

**Prerequisito:** antes de usar esta guía, reejecutar `05_features.ipynb` y `07_calibration.ipynb` con la versión que incluye `hours_since_last_fault`. Los pkl generados sin esa feature no son compatibles con estos scripts.

El objetivo es simular el despliegue real de un sistema de mantenimiento predictivo:
la Turbina 1 tiene un modelo entrenado con 5 años de histórico. La Turbina 2 es "nueva"
— empieza sin histórico propio y hereda el conocimiento de la Turbina 1.

El sistema funciona en tres ciclos:

| Ciclo | Frecuencia | Qué hace |
|-------|-----------|----------|
| **Inferencia diaria** | Cada día | Lee los últimos 7 días de telemetría de T2, calcula features, predice horas hasta el próximo fallo de cada familia, genera alertas si procede |
| **Reentrenamiento mensual** | Cada mes | Une los nuevos datos de T2 con el histórico de T1, reentrena los modelos, actualiza los pkl |
| **Actualización de fallos** | Mensual (junto al reentrenamiento) | El técnico registra los fallos reales ocurridos en T2 ese mes — se añaden al catálogo de eventos |

---

## Decisión sobre las Fechas de T2

**Sí, cambia las fechas de T2 para que empiecen en 2026.**

El script de limpieza (`kelmarsh_cleaning_script.py`) genera
`turbine_2_telemetry_clean.parquet` con las fechas originales del CSV.
Añade un paso de desplazamiento temporal: los datos de T2 (2018-2022)
se desplazan a 2026-2030, de forma que "hoy" corresponda siempre a
un punto realista de la serie.

El desplazamiento se hace una sola vez, antes de cualquier otro paso.
La lógica es: `timestamp_nuevo = timestamp_original + (fecha_inicio_simulacion - fecha_min_dataset)`.

---

## Estructura de Archivos del Proyecto

```
ai-driven/
├── data/
│   ├── bronze/                         ← CSVs originales (nunca se tocan)
│   ├── silver/
│   │   ├── turbine_1_telemetry_clean.parquet   ← T1, ya generado
│   │   ├── turbine_2_telemetry_clean.parquet   ← T2, generado por kelmarsh_cleaning_script.py
│   │   ├── fault_targets_grouped.parquet        ← fallos de T1, ya generado
│   │   ├── turbine_2_fault_log.csv              ← fallos reales de T2 (se va rellenando)
│   │   ├── features_yaw_cable.parquet           ← T1, ya generados
│   │   ├── features_generator.parquet
│   │   ├── features_brake_hydro.parquet
│   │   ├── features_pitch_bat.parquet
│   │   └── turbine_2_features_window.parquet    ← ventana rolling de T2 (7 días, actualización diaria)
│   └── models/
│       ├── model_yaw_cable.pkl                  ← pipeline {lgbm, calibrator, feature_cols, cfg}
│       ├── model_generator.pkl
│       ├── model_brake_hydro.pkl
│       ├── model_pitch_bat.pkl
│       ├── results_*.json
│       └── predictions_log.csv                  ← histórico de predicciones diarias de T2
├── src/
│   ├── t2_00_shift_timestamps.py        ← PASO 0: desplazar fechas de T2
│   ├── t2_01_feature_engineering.py     ← PASO 1: calcular features para T2
│   ├── t2_02_daily_inference.py         ← PASO 2: inferencia diaria (cron diario)
│   ├── t2_03_register_faults.py         ← PASO 3: registrar fallos reales de T2
│   ├── t2_04_monthly_retrain.py         ← PASO 4: reentrenamiento mensual (cron mensual)
│   └── shared/
│       ├── feature_builder.py           ← lógica compartida de features (usada por 01 y 04)
│       └── model_loader.py              ← carga pkl, desempaqueta pipeline
├── notebooks/                           ← los notebooks del proyecto T1
├── kelmarsh_cleaning_script.py          ← script de limpieza con PySpark (ya existe)
└── logs/
```

---

## PASO 0 — Desplazar timestamps de T2 a 2026

**Archivo:** `src/t2_00_shift_timestamps.py`

**Cuándo ejecutarlo:** Una sola vez, antes de todo lo demás.

**Qué hace:**
1. Lee `turbine_2_telemetry_clean.parquet`
2. Calcula el offset: `2026-01-01 00:00 - timestamp_mínimo_del_dataset`
3. Suma el offset a todos los timestamps
4. Guarda el resultado sobreescribiendo el mismo archivo (o como `_shifted`)

**Por qué:** Los datos de T2 son de 2018-2022 igual que T1. Si usamos las fechas
originales, "hoy" (2026) no coincide con ningún dato y no hay forma de simular
la consulta diaria. Con el desplazamiento, el 1 de enero de 2026 corresponde
al primer registro de T2, y cada día del script de inferencia avanza un día en T2.

```python
# src/t2_00_shift_timestamps.py
import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SILVER_DIR = os.path.join(BASE_DIR, 'data', 'silver')
SIM_START = pd.Timestamp('2026-01-01')   # fecha de inicio de la simulación

def shift_timestamps():
    path = os.path.join(SILVER_DIR, 'turbine_2_telemetry_clean.parquet')
    df = pd.read_parquet(path)
    df = df.sort_values('timestamp').reset_index(drop=True)

    t_min = df['timestamp'].min()
    offset = SIM_START - t_min
    print(f'Offset de desplazamiento: {offset}')
    print(f'Rango original:   {t_min}  →  {df["timestamp"].max()}')

    df['timestamp'] = df['timestamp'] + offset
    print(f'Rango desplazado: {df["timestamp"].min()}  →  {df["timestamp"].max()}')

    out = os.path.join(SILVER_DIR, 'turbine_2_telemetry_clean.parquet')
    df.to_parquet(out, index=False)
    print(f'✅ Guardado: {out}')

if __name__ == '__main__':
    shift_timestamps()
```

---

## PASO 1 — Calcular Features de T2 (primera vez + ventana diaria)

**Archivo:** `src/t2_01_feature_engineering.py`
**Archivo compartido:** `src/shared/feature_builder.py`

**Cuándo ejecutarlo:** Una vez para generar el histórico inicial,
luego se llama internamente desde el script de inferencia diaria.

### Por qué T2 necesita su propio cálculo de features

Los modelos de T1 se entrenaron con features calculadas sobre T1.
Para predecir en T2, hay que aplicar **exactamente las mismas transformaciones**
(mismas ventanas, mismo baseline, mismas features de dominio).

**Importante:** el baseline (media y p90 de los primeros 180 días) debe calcularse
sobre los primeros 180 días de T2, no de T1. Cada turbina tiene su propio estado
"sano" de referencia.

### `src/shared/feature_builder.py`

Este módulo centraliza toda la lógica de features para que sea idéntica
entre el entrenamiento de T1 y la inferencia de T2.

```python
# src/shared/feature_builder.py
"""
M�dulo compartido de feature engineering.
Usado por:
  - t2_01_feature_engineering.py  (generar features de T2)
  - t2_04_monthly_retrain.py      (reentrenar con T1+T2)
"""
import pandas as pd
import numpy as np

WINDOWS = {'1h': 6, '6h': 36, '24h': 144, '7d': 1008}
BASELINE_DAYS = 180

FAMILY_SENSORS = {
    'yaw_cable': [
        'nacelle_position', 'nacelle_position_standard_deviation',
        'wind_direction', 'wind_direction_standard_deviation',
        'vane_position_12', 'cable_windings_from_calibration_point',
        'wind_speed_ms', 'power_kw',
        'yaw_error', 'yaw_error_wind', 'cable_rate', 'nacelle_std_ratio',
    ],
    'brake_hydro': [
        'gear_oil_inlet_pressure_bar', 'gear_oil_pump_pressure_bar',
        'gear_oil_inlet_temperature_c', 'gear_oil_temperature_c',
        'generator_rpm_rpm', 'generator_rpm_standard_deviation_rpm',
        'rotor_speed_rpm', 'power_kw',
        'front_bearing_temperature_c', 'rear_bearing_temperature_c',
        'metal_particle_count',
        't_gear_oil_delta', 'pressure_vs_temp', 'metal_particle_rate',
    ],
    'generator': [
        'generator_bearing_front_temperature_c', 'generator_bearing_rear_temperature_c',
        'generator_bearing_front_temperature_max_c', 'generator_bearing_rear_temperature_max_c',
        'nacelle_temperature_c', 'nacelle_ambient_temperature_c',
        'ambient_temperature_converter_c', 'power_kw', 'reactive_power_kvar',
        'power_factor_cosphi', 'stator_temperature_1_c', 'wind_speed_ms',
        't_bearing_delta', 't_rear_bearing_delta', 't_stator_delta',
        't_bearing_diff', 't_stator_bearing_diff',
        'apparent_power_kva', 'reactive_power_ratio',
        't_bearing_delta_roc', 't_stator_roc',
    ],
    'pitch_bat': [
        'motor_current_axis_1_a', 'motor_current_axis_2_a', 'motor_current_axis_3_a',
        'blade_angle_pitch_position_a', 'blade_angle_pitch_position_b', 'blade_angle_pitch_position_c',
        't_motor1_vs_ambient', 't_motor2_vs_ambient', 't_motor3_vs_ambient',
        'power_kw', 'wind_speed_ms',
        'pitch_asymmetry', 'blade_angle_mean', 'motor_current_imbalance',
    ],
}

def add_domain_features(df: pd.DataFrame) -> pd.DataFrame:
    """Añade features calculadas de dominio físico."""
    df = df.copy()
    df['yaw_error']         = (df['nacelle_position'] - df['wind_direction']).abs() % 360
    df['yaw_error']         = df['yaw_error'].apply(lambda x: x if x <= 180 else 360 - x)
    df['yaw_error_wind']    = df['yaw_error'] * df['wind_speed_ms']
    df['cable_rate']        = df['cable_windings_from_calibration_point'].diff(1).fillna(0)
    df['nacelle_std_ratio'] = df['nacelle_position_standard_deviation'] / (df['wind_speed_ms'] + 1e-6)
    df['t_bearing_delta']      = df['generator_bearing_front_temperature_c'] - df['nacelle_ambient_temperature_c']
    df['t_rear_bearing_delta'] = df['generator_bearing_rear_temperature_c']  - df['nacelle_ambient_temperature_c']
    df['t_stator_delta']       = df['stator_temperature_1_c']                - df['nacelle_ambient_temperature_c']
    df['t_gear_oil_delta']     = df['gear_oil_temperature_c']                - df['nacelle_ambient_temperature_c']
    df['t_bearing_diff']       = df['generator_bearing_front_temperature_c'] - df['generator_bearing_rear_temperature_c']
    df['t_stator_bearing_diff']= df['stator_temperature_1_c']                - df['generator_bearing_front_temperature_c']
    df['t_bearing_delta_roc']  = df['t_bearing_delta'].diff(6)
    df['t_stator_roc']         = df['stator_temperature_1_c'].diff(6)
    df['apparent_power_kva']   = (df['power_kw']**2 + df['reactive_power_kvar']**2) ** 0.5
    df['reactive_power_ratio'] = df['reactive_power_kvar'] / (df['apparent_power_kva'] + 1e-6)
    df['pressure_vs_temp']     = df['gear_oil_inlet_pressure_bar'] / (df['gear_oil_inlet_temperature_c'] + 273.15)
    df['metal_particle_rate']  = df['metal_particle_count'].diff(1).fillna(0).clip(lower=0)
    df['t_motor1_vs_ambient']  = df['temperature_motor_axis_1_c'] - df['nacelle_ambient_temperature_c']
    df['t_motor2_vs_ambient']  = df['temperature_motor_axis_2_c'] - df['nacelle_ambient_temperature_c']
    df['t_motor3_vs_ambient']  = df['temperature_motor_axis_3_c'] - df['nacelle_ambient_temperature_c']
    df['pitch_asymmetry']      = (df[['blade_angle_pitch_position_a','blade_angle_pitch_position_b','blade_angle_pitch_position_c']].max(axis=1) -
                                   df[['blade_angle_pitch_position_a','blade_angle_pitch_position_b','blade_angle_pitch_position_c']].min(axis=1))
    df['blade_angle_mean']         = df[['blade_angle_pitch_position_a','blade_angle_pitch_position_b','blade_angle_pitch_position_c']].mean(axis=1)
    df['motor_current_imbalance']  = df[['motor_current_axis_1_a','motor_current_axis_2_a','motor_current_axis_3_a']].std(axis=1)
    return df

def compute_baseline(df: pd.DataFrame) -> tuple[dict, dict]:
    """Calcula baseline (mean y p90) sobre los primeros 180 días."""
    cutoff = df['timestamp'].min() + pd.Timedelta(days=BASELINE_DAYS)
    df_bl  = df[df['timestamp'] < cutoff]
    sensor_cols = [c for c in df.columns
                   if c not in ['timestamp'] and not c.startswith('is_pre_')
                   and not c.startswith('hours_to_') and not c.startswith('hours_since_')
                   and df[c].dtype in [float, 'float64', 'float32']]
    return df_bl[sensor_cols].mean().to_dict(), df_bl[sensor_cols].quantile(0.90).to_dict()

def make_rolling_features(df: pd.DataFrame, sensors: list,
                           baseline_mean: dict, baseline_p90: dict) -> pd.DataFrame:
    """Genera features rolling: mean, std, p95, exceedance, baseline_ratio."""
    feats = {}
    for col in sensors:
        if col not in df.columns:
            continue
        s      = df[col].ffill().fillna(0)
        thresh = baseline_p90.get(col, s.quantile(0.90))
        for wname, w in WINDOWS.items():
            mp   = max(1, w // 3)
            roll = s.rolling(w, min_periods=mp)
            feats[f'{col}__mean_{wname}']   = roll.mean()
            feats[f'{col}__std_{wname}']    = roll.std().fillna(0)
            feats[f'{col}__p95_{wname}']    = roll.quantile(0.95)
            feats[f'{col}__exceed_{wname}'] = s.rolling(w, min_periods=mp).apply(
                lambda x: (x > thresh).mean(), raw=True)
        bm = baseline_mean.get(col, 1.0)
        if abs(bm) > 1e-6:
            feats[f'{col}__baseline_ratio'] = s.rolling(
                WINDOWS['7d'], min_periods=max(1, WINDOWS['7d']//3)
            ).mean() / abs(bm)
    return pd.DataFrame(feats, index=df.index)

def add_temporal_context(df: pd.DataFrame, family: str, fault_times: list) -> pd.DataFrame:
    """
    Añade hours_since_last_fault y su versión log.
    fault_times: fallos PASADOS conocidos (no futuros — sin data leakage).
    En T2 sin histórico propio, se pasan los fallos de T1 como aproximación inicial.
    """
    if not fault_times:
        # Sin fallos de referencia: asignar 1 año a todas las filas
        import pandas as _pd
        df[f"hours_since_last_{family}"]     = 8760.0
        df[f"hours_since_last_{family}_log"] = float(np.log1p(8760.0))
        return df
    fault_arr   = np.array(fault_times, dtype='datetime64[ns]')
    ts_arr      = df['timestamp'].values.astype('datetime64[ns]')
    hours_since = np.full(len(ts_arr), np.nan)
    for i, ts in enumerate(ts_arr):
        past = fault_arr[fault_arr <= ts]
        hours_since[i] = float((ts - past[-1]) / np.timedelta64(1,'h')) if len(past) > 0 else 8760.0
    df[f'hours_since_last_{family}']     = hours_since
    df[f'hours_since_last_{family}_log'] = np.log1p(hours_since)
    return df
```

### `src/t2_01_feature_engineering.py`

```python
# src/t2_01_feature_engineering.py
"""
Genera features_turbine2_{familia}.parquet para todo el histórico de T2.
Ejecutar una sola vez tras el PASO 0.
"""
import os, json, time
import pandas as pd
import numpy as np
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'shared'))
from feature_builder import (
    FAMILY_SENSORS, add_domain_features,
    compute_baseline, make_rolling_features, add_temporal_context
)

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SILVER_DIR = os.path.join(BASE_DIR, 'data', 'silver')
TURBINE_ID = 2

def main():
    print('Cargando telemetría T2...')
    df = pd.read_parquet(os.path.join(SILVER_DIR, f'turbine_{TURBINE_ID}_telemetry_clean.parquet'))
    df = df.sort_values('timestamp').reset_index(drop=True)
    print(f'  {len(df):,} filas  |  {df["timestamp"].min().date()} → {df["timestamp"].max().date()}')

    df = add_domain_features(df)
    baseline_mean, baseline_p90 = compute_baseline(df)

    # Guardar baseline de T2 (necesario para inferencia diaria)
    baseline_path = os.path.join(SILVER_DIR, f'turbine_{TURBINE_ID}_baseline.json')
    json.dump({'mean': baseline_mean, 'p90': baseline_p90}, open(baseline_path, 'w'), default=str)
    print(f'  Baseline guardado: {baseline_path}')

    # Cargar log de fallos de T2 (vacío al inicio, se va rellenando)
    fault_log_path = os.path.join(SILVER_DIR, f'turbine_{TURBINE_ID}_fault_log.csv')
    if os.path.exists(fault_log_path):
        fault_log = pd.read_csv(fault_log_path, parse_dates=['timestamp'])
    else:
        # Sin fallos aún — crear vacío con estructura correcta
        fault_log = pd.DataFrame(columns=['timestamp', 'family', 'code', 'message'])
        fault_log.to_csv(fault_log_path, index=False)
        print(f'  Log de fallos vacío creado: {fault_log_path}')

    for family, sensors in FAMILY_SENSORS.items():
        t0 = time.time()
        print(f'\n  Calculando features: {family}...')
        feats = make_rolling_features(df, sensors, baseline_mean, baseline_p90)

        # Contexto temporal: usar fallos de T1 al inicio, T2 cuando los haya
        fault_times = fault_log[fault_log['family'] == family]['timestamp'].tolist()
        if not fault_times:
            # Sin fallos de T2 aún: usar fallos de T1 como contexto histórico
            t1_targets = pd.read_parquet(os.path.join(SILVER_DIR, 'fault_targets_grouped.parquet'))
            fault_times = t1_targets[t1_targets['family'] == family]['timestamp'].tolist()
            print(f'    ℹ️  Usando {len(fault_times)} fallos de T1 como contexto inicial')
        feats = add_temporal_context(pd.concat([df[['timestamp']], feats], axis=1),
                                     family, fault_times)

        out_path = os.path.join(SILVER_DIR, f'turbine_{TURBINE_ID}_features_{family}.parquet')
        pd.concat([df[['timestamp']], feats], axis=1).to_parquet(out_path, index=False)
        print(f'  ✅ {out_path}  [{time.time()-t0:.0f}s]')

if __name__ == '__main__':
    main()
```

---

## PASO 2 — Inferencia Diaria

**Archivo:** `src/t2_02_daily_inference.py`

**Cuándo ejecutarlo:** Todos los días a la misma hora mediante cron o EventBridge.

**Qué hace exactamente:**

1. Calcula qué fecha simula "hoy" (la última fecha disponible en los datos de T2 que sea ≤ hoy real)
2. Extrae los **últimos 7 días** de telemetría de T2 hasta esa fecha
3. Aplica las features de dominio y rolling sobre esa ventana
4. Carga cada `model_{familia}.pkl` y predice `hours_to_fault` para cada fila
5. Para cada familia, reporta el mínimo predicho (el momento más cercano al fallo según el modelo)
6. Si el mínimo está por debajo del umbral de alerta, emite una alerta
7. Guarda la predicción del día en `predictions_log.csv`

**Por qué solo los últimos 7 días:**
Las ventanas rolling más largas son de 7 días (1008 pasos de 10 min).
Para que estén completamente calculadas, necesitas exactamente 7 días de historia.
No necesitas cargar todo el dataset completo en cada ejecución diaria.

```python
# src/t2_02_daily_inference.py
"""
Inferencia diaria sobre Turbina 2.
Ejecutar cada día mediante cron o EventBridge.
Produce UNA fila en predictions_log.csv por familia con la predicción del día.

Lógica de fechas:
  - "Hoy" = fecha real del sistema en el momento de ejecución (pd.Timestamp.now())
  - El dataset de T2 tiene fechas desplazadas a 2026+ (PASO 0)
  - El script solo lee datos con timestamp <= hoy, simulando que los datos
    futuros aún no han llegado — exactamente como en producción real
  - Para calcular las features de ventana 7d correctamente, necesita los
    últimos 7 días de datos hasta hoy (1008 pasos de 10min)
  - La predicción final es sobre UNA SOLA FILA: el último instante disponible
"""
import os, json, pickle, logging
from datetime import datetime
import pandas as pd
import numpy as np
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'shared'))
from feature_builder import (
    FAMILY_SENSORS, add_domain_features,
    make_rolling_features, add_temporal_context
)

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SILVER_DIR = os.path.join(BASE_DIR, 'data', 'silver')
MODELS_DIR = os.path.join(BASE_DIR, 'data', 'models')
TURBINE_ID = 2

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

FAMILIES = {
    'yaw_cable':   {'alert_h': 48,  'lead_hours': 72},
    'generator':   {'alert_h': 72,  'lead_hours': 120},
    'brake_hydro': {'alert_h': 72,  'lead_hours': 120},
    'pitch_bat':   {'alert_h': 168, 'lead_hours': 336},
}

# Pasos de 10min en 7 días = 7 * 24 * 6 = 1008
# Se cargan exactamente estos pasos para que las ventanas 7d estén completas
STEPS_7D = 1008

def run_daily_inference():
    logger.info('=' * 60)
    logger.info('INFERENCIA DIARIA — Turbina %d', TURBINE_ID)
    logger.info('=' * 60)

    # "Hoy" es la fecha real del sistema.
    # Como el dataset de T2 tiene fechas desplazadas a 2026+,
    # esto funciona igual que en producción real con datos en streaming.
    hoy = pd.Timestamp.now().normalize()
    logger.info('Fecha de hoy: %s', hoy.date())

    # Cargar telemetría de T2 completa y filtrar solo hasta hoy.
    # Esto simula que los datos futuros aún no han llegado.
    df_all = pd.read_parquet(
        os.path.join(SILVER_DIR, f'turbine_{TURBINE_ID}_telemetry_clean.parquet')
    ).sort_values('timestamp').reset_index(drop=True)

    df_hasta_hoy = df_all[df_all['timestamp'] <= hoy].copy()

    if len(df_hasta_hoy) == 0:
        logger.error('Sin datos hasta %s — verifica que el dataset de T2 '
                     'tiene fechas desplazadas a 2026+ (ejecuta t2_00_shift_timestamps.py)', hoy.date())
        return

    # Tomar los últimos STEPS_7D pasos (7 días de historia).
    # Son los necesarios para calcular correctamente las features de ventana 7d.
    # Si hay menos de 1008 filas (primeros días de operación), se usan todas las disponibles.
    df_window = df_hasta_hoy.tail(STEPS_7D).reset_index(drop=True)

    ultimo_ts = df_window['timestamp'].max()
    primer_ts = df_window['timestamp'].min()
    logger.info('Último dato disponible: %s', ultimo_ts)
    logger.info('Ventana cargada: %s → %s (%d filas)',
                primer_ts.date(), ultimo_ts.date(), len(df_window))

    if len(df_window) < STEPS_7D:
        logger.warning('Menos de 7 días de datos disponibles (%d filas). '
                       'Las features _7d estarán parcialmente calculadas.', len(df_window))

    # Añadir features de dominio (yaw_error, t_bearing_delta, etc.)
    df_window = add_domain_features(df_window)

    # Cargar baseline de T2 (calculado en PASO 1, fijo para toda la vida del modelo)
    with open(os.path.join(SILVER_DIR, f'turbine_{TURBINE_ID}_baseline.json')) as f:
        bl = json.load(f)
    baseline_mean = bl['mean']
    baseline_p90  = bl['p90']

    # Cargar log de fallos de T2 (se va rellenando con t2_03_register_faults.py)
    fault_log_path = os.path.join(SILVER_DIR, f'turbine_{TURBINE_ID}_fault_log.csv')
    fault_log = pd.read_csv(fault_log_path, parse_dates=['timestamp'])         if os.path.exists(fault_log_path)         else pd.DataFrame(columns=['timestamp', 'family'])

    pred_log_path  = os.path.join(MODELS_DIR, 'predictions_log.csv')
    results_today  = []

    for family, cfg in FAMILIES.items():
        logger.info('  Familia: %s', family)

        model_path = os.path.join(MODELS_DIR, f'model_{family}.pkl')
        if not os.path.exists(model_path):
            logger.warning('  Modelo no encontrado: %s', model_path)
            continue

        pipeline     = pickle.load(open(model_path, 'rb'))
        lgbm         = pipeline['lgbm']
        calibrator   = pipeline['calibrator']
        feature_cols = pipeline['feature_cols']

        # Calcular features rolling sobre la ventana de 7 días
        feats = make_rolling_features(
            df_window, FAMILY_SENSORS[family], baseline_mean, baseline_p90
        )

        # Contexto temporal: fallos propios de T2 si los hay,
        # si no, fallos de T1 como aproximación inicial
        fault_times = fault_log[fault_log['family'] == family]['timestamp'].tolist()
        if not fault_times:
            t1_targets  = pd.read_parquet(
                os.path.join(SILVER_DIR, 'fault_targets_grouped.parquet')
            )
            fault_times = t1_targets[
                t1_targets['family'] == family
            ]['timestamp'].tolist()
            logger.info('  Contexto T1 (%d fallos) — T2 sin histórico propio aún',
                        len(fault_times))

        df_feats = pd.concat([df_window[['timestamp']], feats], axis=1)
        df_feats = add_temporal_context(df_feats, family, fault_times)

        # Añadir columnas ausentes (por si algún sensor no estaba en la ventana)
        for col in feature_cols:
            if col not in df_feats.columns:
                df_feats[col] = 0.0

        # PREDICCIÓN SOBRE UNA SOLA FILA: el último instante disponible (hoy)
        # Los 7 días anteriores solo servían para calcular las features correctamente.
        X_hoy = df_feats[feature_cols].fillna(0).iloc[[-1]]

        raw_pred = float(np.clip(lgbm.predict(X_hoy)[0], 0, cfg['lead_hours']))
        cal_pred = float(np.clip(calibrator.predict([raw_pred])[0], 0, cfg['lead_hours']))

        is_alert = cal_pred <= cfg['alert_h']

        logger.info('  Predicción: %.1fh  |  Umbral alerta: %dh  |  Alerta: %s',
                    cal_pred, cfg['alert_h'], '🚨 SÍ' if is_alert else '✅ NO')

        if is_alert:
            logger.warning('  🚨 ALERTA — %s: fallo predicho en %.1fh', family, cal_pred)

        results_today.append({
            'date':              str(hoy.date()),
            'last_data_ts':      str(ultimo_ts),
            'turbine':           TURBINE_ID,
            'family':            family,
            'pred_h':            round(cal_pred, 1),
            'alert':             is_alert,
            'alert_threshold_h': cfg['alert_h'],
        })

    # Guardar en log acumulativo
    df_results = pd.DataFrame(results_today)
    write_header = not os.path.exists(pred_log_path)
    df_results.to_csv(pred_log_path, mode='a', header=write_header, index=False)
    logger.info('Predicciones guardadas: %s', pred_log_path)

    print('\n' + '='*60)
    print(f'RESUMEN — {hoy.date()}  (último dato: {ultimo_ts.date()})')
    print('='*60)
    for r in results_today:
        status = '🚨 ALERTA' if r['alert'] else '✅ Normal'
        print(f'  {r["family"]:<15}: {r["pred_h"]:>6.1f}h  {status}')

if __name__ == '__main__':
    run_daily_inference()

---

## PASO 3 — Registrar Fallos Reales de T2

**Archivo:** `src/t2_03_register_faults.py`

**Cuándo ejecutarlo:** Manualmente, cuando ocurre un fallo real en T2
que el técnico confirma (o automáticamente si tienes el SCADA en tiempo real).

**Qué hace:**
Añade el fallo al log `turbine_2_fault_log.csv`. Este log se usa
en dos sitios: el contexto temporal de la inferencia diaria (para que el modelo
sepa cuánto tiempo ha pasado desde el último fallo) y el reentrenamiento mensual.

```python
# src/t2_03_register_faults.py
"""
Registra un fallo real ocurrido en Turbina 2.
Uso: python t2_03_register_faults.py --family generator --timestamp "2026-03-15 14:20" --code 3000
"""
import os, argparse, csv
from datetime import datetime
import pandas as pd

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SILVER_DIR = os.path.join(BASE_DIR, 'data', 'silver')

VALID_FAMILIES = ['yaw_cable', 'generator', 'brake_hydro', 'pitch_bat']

FAULT_FAMILIES_CODES = {
    'yaw_cable':   [6052, 6200, 6054, 6120, 6300],
    'brake_hydro': [2125, 5720, 5510, 2000, 1860],
    'generator':   [3000, 2550, 2650, 2655, 2674, 8400, 3125],
    'pitch_bat':   [716, 717, 718, 681, 682, 683, 785, 850],
}

def register_fault(family: str, timestamp: str, code: int, message: str = ''):
    if family not in VALID_FAMILIES:
        raise ValueError(f'Familia inválida: {family}. Válidas: {VALID_FAMILIES}')

    fault_log_path = os.path.join(SILVER_DIR, 'turbine_2_fault_log.csv')
    ts = pd.Timestamp(timestamp)

    row = {
        'timestamp': ts.floor('10min'),   # redondear al intervalo SCADA
        'family':    family,
        'code':      code,
        'message':   message,
        'registered_at': datetime.now().isoformat(),
    }

    file_exists = os.path.exists(fault_log_path)
    with open(fault_log_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

    print(f'✅ Fallo registrado:')
    print(f'   Familia:   {family}')
    print(f'   Timestamp: {row["timestamp"]}')
    print(f'   Código:    {code}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--family',    required=True)
    parser.add_argument('--timestamp', required=True)
    parser.add_argument('--code',      required=True, type=int)
    parser.add_argument('--message',   default='')
    args = parser.parse_args()
    register_fault(args.family, args.timestamp, args.code, args.message)
```

---

## PASO 4 — Reentrenamiento Mensual

**Archivo:** `src/t2_04_monthly_retrain.py`

**Cuándo ejecutarlo:** El primer día de cada mes, mediante cron o EventBridge.

**Qué hace exactamente:**

1. Carga `features_{familia}.parquet` de T1 (el histórico de entrenamiento original)
2. Construye las features de T2 para **todos los datos acumulados hasta la fecha** (aplicando el mismo pipeline del `feature_builder.py`)
3. Etiqueta los datos de T2 con `hours_to_fault` usando los fallos registrados en `turbine_2_fault_log.csv`
4. **Concatena** T1 + T2 etiquetados — ese es el nuevo conjunto de entrenamiento
5. Reentrena con el mismo pipeline del `07_calibration.ipynb` (LGBMRegressor + IsotonicRegression)
6. Guarda los nuevos `model_{familia}.pkl` y `results_{familia}.json`
7. Registra en `retrain_log.csv` qué fecha, cuántos datos de T2 se añadieron, y métricas antes/después

**Por qué unir T1+T2 y no solo entrenar con T2:**
T2 lleva pocos meses en producción — no tiene suficiente histórico de fallos para
aprender patrones. Al unir T1+T2, el modelo sigue aprendiendo del rico histórico
de T1 y se adapta progresivamente a las particularidades de T2. Con el tiempo,
cuando T2 acumule 2-3 años de datos propios, el peso de T1 irá siendo menor.

**Sobre el etiquetado de T2:**
Las features de T2 no tienen `hours_to_fault` porque ese valor se calcula
a partir de los fallos reales registrados en `turbine_2_fault_log.csv`.
Si no hay fallos registrados de T2, solo se usa T1 para el reentrenamiento.

```python
# src/t2_04_monthly_retrain.py
"""
Reentrenamiento mensual: combina histórico T1 + datos acumulados T2.
Ejecutar el primer día de cada mes.
"""
import os, json, pickle, logging, csv
from datetime import datetime
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import mean_absolute_error
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'shared'))
from feature_builder import (
    FAMILY_SENSORS, add_domain_features,
    make_rolling_features, add_temporal_context
)

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SILVER_DIR = os.path.join(BASE_DIR, 'data', 'silver')
MODELS_DIR = os.path.join(BASE_DIR, 'data', 'models')
TURBINE_ID = 2

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

FAMILIES = {
    'yaw_cable':   {'lead_hours': 72,  'alert_h': 48},
    'generator':   {'lead_hours': 120, 'alert_h': 72},
    'brake_hydro': {'lead_hours': 120, 'alert_h': 72},
    'pitch_bat':   {'lead_hours': 336, 'alert_h': 168},
}

LGBM_PARAMS = {
    'yaw_cable':   {'n_estimators':1000,'learning_rate':0.05,'num_leaves':63, 'min_child_samples':20,'subsample':0.8,'colsample_bytree':0.8,'reg_alpha':0.1,'reg_lambda':0.1,'random_state':42,'n_jobs':-1,'verbose':-1},
    'generator':   {'n_estimators':1000,'learning_rate':0.05,'num_leaves':63, 'min_child_samples':20,'subsample':0.8,'colsample_bytree':0.8,'reg_alpha':0.1,'reg_lambda':0.1,'random_state':42,'n_jobs':-1,'verbose':-1},
    'brake_hydro': {'n_estimators':1000,'learning_rate':0.05,'num_leaves':31, 'min_child_samples':30,'subsample':0.8,'colsample_bytree':0.8,'reg_alpha':0.1,'reg_lambda':0.1,'random_state':42,'n_jobs':-1,'verbose':-1},
    'pitch_bat':   {'n_estimators':1000,'learning_rate':0.05,'num_leaves':63, 'min_child_samples':20,'subsample':0.8,'colsample_bytree':0.8,'reg_alpha':0.1,'reg_lambda':0.1,'random_state':42,'n_jobs':-1,'verbose':-1},
}

def label_family_t2(df_t2: pd.DataFrame, fault_log: pd.DataFrame,
                     family: str, lead_hours: int) -> pd.DataFrame:
    """
    Etiqueta los datos de T2 con hours_to_fault y is_pre_fault.
    Requiere que fault_log tenga fallos reales de T2 para esta familia.
    """
    fault_times = sorted(
        fault_log[fault_log['family'] == family]['timestamp'].tolist()
    )
    if not fault_times:
        logger.warning('  Sin fallos de T2 para %s — no se añade T2 al entrenamiento', family)
        return None

    df = df_t2.copy().sort_values('timestamp').reset_index(drop=True)
    ts_arr    = df['timestamp'].values.astype('datetime64[ns]')
    fault_arr = np.array(fault_times, dtype='datetime64[ns]')
    hours_arr = np.full(len(ts_arr), np.nan)

    for i, ts in enumerate(ts_arr):
        future = fault_arr[fault_arr > ts]
        if len(future) == 0:
            continue
        delta_h = (future[0] - ts) / np.timedelta64(1, 'h')
        if delta_h <= lead_hours:
            hours_arr[i] = delta_h

    df[f'hours_to_{family}'] = hours_arr
    df[f'is_pre_{family}']   = ~np.isnan(hours_arr)
    return df

def run_monthly_retrain():
    logger.info('=' * 60)
    logger.info('REENTRENAMIENTO MENSUAL — %s', datetime.now().strftime('%Y-%m-%d'))
    logger.info('=' * 60)

    # Cargar fallos de T2
    fault_log_path = os.path.join(SILVER_DIR, 'turbine_2_fault_log.csv')
    if os.path.exists(fault_log_path):
        fault_log_t2 = pd.read_csv(fault_log_path, parse_dates=['timestamp'])
        logger.info('Fallos T2 registrados: %d', len(fault_log_t2))
    else:
        fault_log_t2 = pd.DataFrame(columns=['timestamp', 'family', 'code', 'message'])
        logger.warning('Sin fallos de T2 registrados — reentrenamiento solo con T1')

    # Cargar telemetría de T2 y calcular features
    df_t2_raw = pd.read_parquet(
        os.path.join(SILVER_DIR, f'turbine_{TURBINE_ID}_telemetry_clean.parquet')
    ).sort_values('timestamp').reset_index(drop=True)
    df_t2_raw = add_domain_features(df_t2_raw)

    # Baseline de T2
    with open(os.path.join(SILVER_DIR, f'turbine_{TURBINE_ID}_baseline.json')) as f:
        bl = json.load(f)

    retrain_summary = []

    for family, cfg in FAMILIES.items():
        logger.info('\n  FAMILIA: %s', family.upper())

        # --- Cargar datos de T1 ---
        df_t1 = pd.read_parquet(os.path.join(SILVER_DIR, f'features_{family}.parquet'))
        target_col   = f'is_pre_{family}'
        hours_col    = f'hours_to_{family}'
        feature_cols = [c for c in df_t1.columns if c not in ['timestamp', target_col, hours_col]]

        # --- Construir features de T2 ---
        feats_t2 = make_rolling_features(df_t2_raw, FAMILY_SENSORS[family], bl['mean'], bl['p90'])
        fault_times_t2 = fault_log_t2[fault_log_t2['family'] == family]['timestamp'].tolist()
        fault_times_context = fault_times_t2 if fault_times_t2 else \
            pd.read_parquet(os.path.join(SILVER_DIR, 'fault_targets_grouped.parquet')
            ).query(f'family == "{family}"')['timestamp'].tolist()

        df_t2_feats = pd.concat([df_t2_raw[['timestamp']], feats_t2], axis=1)
        df_t2_feats = add_temporal_context(df_t2_feats, family, fault_times_context)

        # --- Etiquetar T2 ---
        df_t2_labeled = label_family_t2(df_t2_feats, fault_log_t2, family, cfg['lead_hours'])

        # --- Unir T1 + T2 ---
        if df_t2_labeled is not None:
            # Alinear columnas: T2 puede no tener todas las columnas de T1
            for col in feature_cols:
                if col not in df_t2_labeled.columns:
                    df_t2_labeled[col] = 0.0
            cols_needed = ['timestamp', target_col, hours_col] + feature_cols
            df_combined = pd.concat([
                df_t1[cols_needed],
                df_t2_labeled[cols_needed],
            ], ignore_index=True).sort_values('timestamp').reset_index(drop=True)
            logger.info('  T1: %d filas  +  T2: %d filas  =  %d total',
                        len(df_t1), len(df_t2_labeled), len(df_combined))
        else:
            df_combined = df_t1.copy()
            logger.info('  Usando solo T1: %d filas', len(df_combined))

        # --- Split 70/10/20 ---
        df_pos = df_combined[df_combined[target_col]].copy()
        if len(df_pos) < 50:
            logger.warning('  Pocos positivos (%d) — saltando', len(df_pos))
            continue

        n = len(df_pos)
        cutoff_train = df_pos['timestamp'].quantile(0.70)
        cutoff_val   = df_pos['timestamp'].quantile(0.80)

        train_pos = df_pos[df_pos['timestamp'] <  cutoff_train]
        val_pos   = df_pos[(df_pos['timestamp'] >= cutoff_train) & (df_pos['timestamp'] < cutoff_val)]
        test_pos  = df_pos[df_pos['timestamp'] >= cutoff_val]

        if len(val_pos) == 0:
            logger.warning('  Val vacío — saltando')
            continue

        X_train, y_train = train_pos[feature_cols].fillna(0), train_pos[hours_col].values
        X_val,   y_val   = val_pos[feature_cols].fillna(0),   val_pos[hours_col].values
        X_test,  y_test  = test_pos[feature_cols].fillna(0),  test_pos[hours_col].values

        # --- Entrenar LGBMRegressor ---
        lgbm_model = lgb.LGBMRegressor(**LGBM_PARAMS[family])
        lgbm_model.fit(X_train, y_train,
                       eval_set=[(X_val, y_val)],
                       callbacks=[lgb.early_stopping(100, verbose=False),
                                  lgb.log_evaluation(0)])

        pred_val  = np.clip(lgbm_model.predict(X_val),  0, cfg['lead_hours'])
        pred_test = np.clip(lgbm_model.predict(X_test), 0, cfg['lead_hours'])

        # --- Calibrar ---
        cal = IsotonicRegression(out_of_bounds='clip', increasing=True)
        cal.fit(pred_val, y_val)
        pred_test_cal = np.clip(cal.predict(pred_test), 0, cfg['lead_hours'])

        mae = mean_absolute_error(y_test, pred_test_cal)
        logger.info('  MAE calibrado: %.1fh  |  Test positivos: %d', mae, len(test_pos))

        # --- Guardar modelo actualizado ---
        pipeline = {'lgbm': lgbm_model, 'calibrator': cal,
                    'feature_cols': feature_cols, 'family': family, 'cfg': cfg}
        pickle.dump(pipeline, open(os.path.join(MODELS_DIR, f'model_{family}.pkl'), 'wb'))

        retrain_summary.append({
            'date':         datetime.now().isoformat(),
            'family':       family,
            'mae_cal_h':    round(mae, 1),
            'n_t1':         len(df_t1),
            'n_t2':         len(df_t2_labeled) if df_t2_labeled is not None else 0,
            'n_combined':   len(df_combined),
        })
        logger.info('  ✅ Modelo actualizado guardado')

    # Guardar log de reentrenamientos
    retrain_log_path = os.path.join(MODELS_DIR, 'retrain_log.csv')
    if retrain_summary:
        df_log = pd.DataFrame(retrain_summary)
        df_log.to_csv(retrain_log_path, mode='a',
                      header=not os.path.exists(retrain_log_path), index=False)
        logger.info('\n✅ Log de reentrenamiento guardado: %s', retrain_log_path)

if __name__ == '__main__':
    run_monthly_retrain()
```

---

## Automatización con Cron (Linux/Mac) o Task Scheduler (Windows)

### Cron (Linux/Mac)

```bash
# Editar crontab: crontab -e

# Inferencia diaria a las 08:00
0 8 * * * cd /ruta/al/proyecto && /ruta/venv/bin/python src/t2_02_daily_inference.py >> logs/inference_$(date +\%Y\%m\%d).log 2>&1

# Reentrenamiento mensual el día 1 a las 02:00
0 2 1 * * cd /ruta/al/proyecto && /ruta/venv/bin/python src/t2_04_monthly_retrain.py >> logs/retrain_$(date +\%Y\%m\%d).log 2>&1
```

### Si usas AWS EventBridge + Lambda o ECS

El script de inferencia diaria ya está diseñado para ejecutarse como tarea
standalone. El mismo patrón que usas en SunSaver-ETL-Platform sirve aquí.

---

## Orden de Ejecución (Primera Vez)

```
1. kelmarsh_cleaning_script.py    ← limpiar T2 (NUMBER_TURBINE = 2)
2. src/t2_00_shift_timestamps.py  ← desplazar fechas de T2 a 2026
3. src/t2_01_feature_engineering.py ← calcular features históricas de T2
4. src/t2_02_daily_inference.py   ← probar inferencia (primera vez)
```

A partir de ahí:
- `t2_02_daily_inference.py` se ejecuta automáticamente cada día
- `t2_03_register_faults.py` se ejecuta manualmente cuando ocurre un fallo
- `t2_04_monthly_retrain.py` se ejecuta automáticamente el primer día de cada mes

---

## Respuesta a tus Dudas Originales

**¿Cuándo se une el log de fallos con la telemetría?**
En el reentrenamiento mensual (`t2_04`). La inferencia diaria no necesita fallos —
solo necesita las features calculadas. Los fallos se usan para crear el target
`hours_to_fault` de T2, que solo se necesita al reentrenar.

**¿La dataset que entra al modelo tiene `hours_to_fault`?**
No — es el target. El modelo lo predice, no lo recibe.

**¿No es data leakage usar `hours_since_last_fault` en producción?**
No. `hours_since` mira al **pasado**: cuántas horas desde el último fallo YA OCURRIDO.
Ese valor siempre se conoce en producción porque los fallos se registran cuando pasan.
Es distinto de `hours_to_fault` (horas HASTA el próximo fallo = futuro = solo en entrenamiento).

**¿Por qué usar fallos de T1 para el contexto inicial de T2?**
Sin histórico propio de T2, `hours_since` sería 8760h constante — un valor
que el modelo nunca vio en entrenamiento. Los fallos de T1 son mejor aproximación.
En cuanto T2 registre su primer fallo real, el sistema usa T2 automáticamente.
No. El modelo en inferencia solo recibe las features (columnas de sensores
procesadas). `hours_to_fault` es el target de entrenamiento — en producción
el modelo predice ese valor, no lo recibe.

**¿Cambiar las fechas de T2 a 2026+?**
Sí, es lo correcto. Sin eso no puedes simular "qué predice el modelo hoy".
El PASO 0 lo hace de forma reproducible y reversible.

**¿Un script o varios?**
Varios, como están aquí. Cada script hace una cosa concreta y se puede
probar, debuggear y programar de forma independiente.