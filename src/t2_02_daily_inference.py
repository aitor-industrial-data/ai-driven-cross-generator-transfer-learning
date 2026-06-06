"""
Inferencia diaria sobre Turbina 2.
Ejecutar cada día (cron/EventBridge/Task Scheduler).
Produce una fila en predictions_log.csv por familia.
"""
import os, json, pickle, logging
from datetime import datetime, date
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
MODELS_DIR = os.path.join(BASE_DIR, 'data', 'models')
TURBINE_ID = 2
WINDOW_DAYS = 8    # 7 días de ventana + 1 día de margen para completar la ventana 7d

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

FAMILIES = {
    'yaw_cable':   {'alert_h': 48,  'lead_hours': 72},
    'generator':   {'alert_h': 72,  'lead_hours': 120},
    'brake_hydro': {'alert_h': 72,  'lead_hours': 120},
    'pitch_bat':   {'alert_h': 168, 'lead_hours': 336},
}

def get_simulated_today(df: pd.DataFrame) -> pd.Timestamp:
    """
    Determina qué fecha corresponde a 'hoy' en la simulación.
    Usa la última fecha del dataset como fecha de simulación
    (o el timestamp real si estuviéramos en producción real con datos en streaming).
    """
    return df['timestamp'].max()

def run_daily_inference():
    logger.info('=' * 60)
    logger.info('INFERENCIA DIARIA — Turbina %d', TURBINE_ID)
    logger.info('=' * 60)

    # Cargar TODA la telemetría de T2 (eficiente: es un parquet columnar)
    df_all = pd.read_parquet(
        os.path.join(SILVER_DIR, f'turbine_{TURBINE_ID}_telemetry_clean.parquet')
    ).sort_values('timestamp').reset_index(drop=True)

    # Calcular "hoy" en la simulación
    sim_today = get_simulated_today(df_all)
    sim_start = sim_today - pd.Timedelta(days=WINDOW_DAYS)
    logger.info('Fecha simulada de hoy: %s', sim_today.date())
    logger.info('Ventana de datos: %s → %s', sim_start.date(), sim_today.date())

    # Extraer ventana de 8 días
    df_window = df_all[df_all['timestamp'] >= sim_start].copy().reset_index(drop=True)
    logger.info('Filas en ventana: %d', len(df_window))

    if len(df_window) < 100:
        logger.warning('⚠️  Pocos datos en ventana — posible hueco en telemetría')

    # Añadir features de dominio
    df_window = add_domain_features(df_window)

    # Cargar baseline de T2 (calculado en PASO 1)
    baseline_path = os.path.join(SILVER_DIR, f'turbine_{TURBINE_ID}_baseline.json')
    with open(baseline_path) as f:
        bl = json.load(f)
    baseline_mean = bl['mean']
    baseline_p90  = bl['p90']

    # Cargar log de fallos de T2
    fault_log_path = os.path.join(SILVER_DIR, f'turbine_{TURBINE_ID}_fault_log.csv')
    fault_log = pd.read_csv(fault_log_path, parse_dates=['timestamp']) \
        if os.path.exists(fault_log_path) else pd.DataFrame(columns=['timestamp','family'])

    # Cargar log de predicciones anteriores
    pred_log_path = os.path.join(MODELS_DIR, 'predictions_log.csv')
    pred_log_exists = os.path.exists(pred_log_path)

    results_today = []

    for family, cfg in FAMILIES.items():
        logger.info('\n  Familia: %s', family)

        # Cargar modelo (pipeline: lgbm + calibrator + feature_cols)
        model_path = os.path.join(MODELS_DIR, f'model_{family}.pkl')
        if not os.path.exists(model_path):
            logger.warning('  Modelo no encontrado: %s', model_path)
            continue

        pipeline = pickle.load(open(model_path, 'rb'))
        lgbm         = pipeline['lgbm']
        calibrator   = pipeline['calibrator']
        feature_cols = pipeline['feature_cols']
        lead_hours   = cfg['lead_hours']

        # Calcular features rolling para la ventana
        sensors = FAMILY_SENSORS[family]
        feats   = make_rolling_features(df_window, sensors, baseline_mean, baseline_p90)

        # Contexto temporal: hours_since_last_fault
        # El modelo fue entrenado CON esta feature — hay que darla en inferencia.
        # Usa fallos reales de T2; si no hay, usa T1 como aproximacion inicial.
        fault_times = fault_log[fault_log['family'] == family]['timestamp'].tolist()
        if not fault_times:
            t1_targets  = pd.read_parquet(os.path.join(SILVER_DIR, 'fault_targets_grouped.parquet'))
            fault_times = t1_targets[t1_targets['family'] == family]['timestamp'].tolist()
            logger.info('  Contexto T1 (%d fallos) hasta que T2 acumule propio', len(fault_times))

        df_feats = pd.concat([df_window[['timestamp']], feats], axis=1)
        df_feats = add_temporal_context(df_feats, family, fault_times)

        # Construir X con exactamente las columnas del modelo
        missing = [c for c in feature_cols if c not in df_feats.columns]
        for mc in missing:
            df_feats[mc] = 0.0  # columna ausente = imputar con 0

        X = df_feats[feature_cols].fillna(0).tail(144)  # últimas 24h (144 pasos × 10min)

        if len(X) == 0:
            logger.warning('  Sin datos para inferencia')
            continue

        # Predicción + calibración
        raw_pred = np.clip(lgbm.predict(X), 0, lead_hours)
        cal_pred = np.clip(calibrator.predict(raw_pred), 0, lead_hours)

        min_pred_h = float(cal_pred.min())
        mean_pred_h = float(cal_pred.mean())
        is_alert = min_pred_h <= cfg['alert_h']

        logger.info('  Pred mín: %.1fh  |  Pred media: %.1fh  |  Alerta: %s',
                    min_pred_h, mean_pred_h, '🚨 SÍ' if is_alert else '✅ NO')

        if is_alert:
            logger.warning('  🚨 ALERTA — %s: fallo predicho en %.1fh (umbral: %dh)',
                           family, min_pred_h, cfg['alert_h'])

        results_today.append({
            'date':        str(sim_today.date()),
            'turbine':     TURBINE_ID,
            'family':      family,
            'min_pred_h':  round(min_pred_h, 1),
            'mean_pred_h': round(mean_pred_h, 1),
            'alert':       is_alert,
            'alert_threshold_h': cfg['alert_h'],
            'n_rows_used': len(X),
        })

    # Guardar en log
    df_results = pd.DataFrame(results_today)
    if pred_log_exists:
        df_results.to_csv(pred_log_path, mode='a', header=False, index=False)
    else:
        df_results.to_csv(pred_log_path, index=False)

    logger.info('\n✅ Predicciones guardadas en: %s', pred_log_path)

    # Resumen en consola
    print('\n' + '='*60)
    print(f'RESUMEN — {sim_today.date()}')
    print('='*60)
    for r in results_today:
        status = '🚨 ALERTA' if r['alert'] else '✅ Normal'
        print(f'  {r["family"]:<15}: {r["min_pred_h"]:>6.1f}h  {status}')

if __name__ == '__main__':
    run_daily_inference()