"""
Inferencia diaria sobre Turbina 2.
Ejecutar cada día mediante cron o EventBridge.
Produce UNA fila en t2_predictions_log.csv por familia con la predicción del día.

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

    pred_log_path  = os.path.join(MODELS_DIR, 't2_predictions_log.csv')
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