"""
t2_monthly_retrain.py
---------------------
Lambda de reentrenamiento mensual. Se ejecuta el día 1 de cada mes.

Flujo:
  1. Carga Feature Stores de T2 por familia (t2_features_{family}.parquet)
  2. Carga fault log de T2 — filtra solo fallos con timestamp <= hoy (sin futuro)
  3. Etiqueta cada Feature Store con hours_to_fault e is_pre_fault
     (floor al intervalo de 10min para alinear con telemetría)
  4. Carga Feature Stores de T1 (t1_features_{family}.parquet) — ya etiquetados
  5. Por familia, entrena DOS versiones:
       A) Solo T2
       B) T1 + T2
  6. Evalúa Event Recall en el test set de T2 para cada versión
  7. Despliega la versión ganadora como t1_model_{family}.pkl en S3
  8. Guarda métricas del reentrenamiento en models/t2_retrain_results.json
"""

import os
import io
import json
import pickle
import logging
import time
import boto3
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import mean_absolute_error
from botocore.exceptions import ClientError

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

BUCKET_NAME = os.environ.get('AWS_S3_BUCKET', 'ai-driven-cross-generator-transfer-learning')
TURBINE_ID  = 2

FAMILIES = {
    'yaw_cable':   {'lead_hours': 83,  'alert_h': 48},
    'generator':   {'lead_hours': 127, 'alert_h': 72},
    'brake_hydro': {'lead_hours': 130, 'alert_h': 72},
    'pitch_bat':   {'lead_hours': 295, 'alert_h': 168},
}

# Split temporal sobre datos de T2
# 60% train | 20% validación (calibración) | 20% test (evaluación)
TRAIN_RATIO = 0.60
VAL_RATIO   = 0.80

# Período de exclusión post-fallo: filas dentro de este margen
# después de un fallo se descartan (sensores en transición post-reparación)
POST_FAULT_EXCLUSION_H = 24

LGBM_PARAMS_BASE = {
    'n_estimators':     1000,
    'learning_rate':    0.05,
    'subsample':        0.8,
    'colsample_bytree': 0.8,
    'reg_alpha':        0.1,
    'reg_lambda':       0.1,
    'random_state':     42,
    'n_jobs':           -1,
    'verbose':          -1,
}
LGBM_PARAMS = {
    'yaw_cable':   {**LGBM_PARAMS_BASE, 'num_leaves': 63, 'min_child_samples': 20},
    'generator':   {**LGBM_PARAMS_BASE, 'num_leaves': 63, 'min_child_samples': 20},
    'brake_hydro': {**LGBM_PARAMS_BASE, 'num_leaves': 31, 'min_child_samples': 30},
    'pitch_bat':   {**LGBM_PARAMS_BASE, 'num_leaves': 63, 'min_child_samples': 20},
}


# =============================================================================
# Utilidades S3
# =============================================================================

def _load_parquet(s3_client, key: str) -> pd.DataFrame | None:
    try:
        response = s3_client.get_object(Bucket=BUCKET_NAME, Key=key)
        return pd.read_parquet(io.BytesIO(response['Body'].read()))
    except ClientError as e:
        if e.response['Error']['Code'] == 'NoSuchKey':
            return None
        raise


def _save_parquet(s3_client, df: pd.DataFrame, key: str) -> None:
    buf = io.BytesIO()
    df.to_parquet(buf, index=False, engine='pyarrow')
    s3_client.put_object(Bucket=BUCKET_NAME, Key=key, Body=buf.getvalue())


def _save_pickle(s3_client, obj, key: str) -> None:
    s3_client.put_object(Bucket=BUCKET_NAME, Key=key, Body=pickle.dumps(obj))


def _save_json(s3_client, obj: dict, key: str) -> None:
    s3_client.put_object(
        Bucket=BUCKET_NAME,
        Key=key,
        Body=json.dumps(obj, indent=2, default=str).encode('utf-8'),
        ContentType='application/json',
    )


# =============================================================================
# Catálogo de eventos SCADA — mismo que process_t2_fault_events.py
# =============================================================================
EVENTS_CSV_KEY = f'bronze/turbine_{TURBINE_ID}_status_2026_2030.csv'
FAULT_LOG_KEY  = f'models/turbine_{TURBINE_ID}_fault_log.csv'

EXCLUDED_CODES = {
    710, 707, 5760, 5700,           # Mantenimiento / pruebas
    3500, 3585, 3590, 6540, 6682,   # Externos / ambientales
    6690, 64, 68, 61, 815, 3570, 3870,
    20, 25, 21, 8000, 210, 7325, 7324,  # Operativos humanos
    6525, 6515, 6530, 6635, 6620, 6622, # Sensores externos
    4510, 4520, 4530, 4540, 4500,       # Estructurales / genéricos
    59, 100, 5000, 440, 2950, 7057,
}

FAMILY_CODES = {
    'yaw_cable':   [6052, 6200, 6054, 6120, 6300, 6005, 6130],
    'brake_hydro': [2125, 5720, 5510, 2000, 1860, 1070, 1065, 1050, 1800, 1620],
    'generator':   [3000, 2550, 2650, 2655, 2674, 3125, 8400, 3110, 3210, 2810],
    'pitch_bat':   [716, 717, 718, 681, 682, 683, 785, 850, 692, 720, 725, 675],
}

CODE_TO_FAMILY = {
    code: family
    for family, codes in FAMILY_CODES.items()
    for code in codes
}


# =============================================================================
# Etiquetado de T2
# =============================================================================

def update_fault_log(s3_client, hoy: pd.Timestamp) -> pd.DataFrame:
    """
    PASO 0 del reentrenamiento mensual.
    Lee el CSV de eventos SCADA desde S3, filtra por timestamp <= hoy,
    procesa los fallos técnicos y actualiza turbine_2_fault_log.csv.
    Idéntica lógica a process_t2_fault_events.py.
    """
    logger.info('Cargando eventos SCADA desde s3://%s/%s', BUCKET_NAME, EVENTS_CSV_KEY)
    try:
        response  = s3_client.get_object(Bucket=BUCKET_NAME, Key=EVENTS_CSV_KEY)
        df_events = pd.read_csv(
            io.BytesIO(response['Body'].read()),
            parse_dates=['Timestamp start', 'Timestamp end'],
            dayfirst=False,
        )
        df_events = df_events.rename(columns={
            'Timestamp start': 'timestamp_start',
            'Timestamp end':   'timestamp_end',
        })
        logger.info('Eventos cargados: %d filas totales.', len(df_events))
    except ClientError as e:
        if e.response['Error']['Code'] == 'NoSuchKey':
            logger.warning('CSV de eventos no encontrado en S3. Fault log no se actualiza.')
            return pd.DataFrame(columns=['timestamp', 'family', 'code', 'message', 'status'])
        raise

    # Filtrar solo eventos hasta hoy (simulación: el CSV tiene datos hasta 2030)
    df_events = df_events[df_events['timestamp_start'] <= hoy].copy()
    logger.info('Eventos hasta %s: %d filas.', hoy.date(), len(df_events))

    # Solo Stop y Warning
    df_events = df_events[df_events['Status'].isin(['Stop', 'Warning'])].copy()

    # Excluir códigos de ruido
    df_events = df_events[~df_events['Code'].isin(EXCLUDED_CODES)].copy()

    # Mapear a familia y descartar sin asignación
    df_events['family'] = df_events['Code'].map(CODE_TO_FAMILY)
    df_events = df_events[df_events['family'].notna()].copy()

    # Timestamp canónico = floor a 10min para alinear con telemetría
    df_events['timestamp'] = pd.to_datetime(df_events['timestamp_start']).dt.floor('10min')

    # Deduplicar: mismo código en mismo bucket de 10min → primero
    df_events = df_events.sort_values('timestamp').reset_index(drop=True)
    df_events = df_events.drop_duplicates(subset=['Code', 'timestamp'], keep='first')

    df_faults = df_events[['timestamp', 'family', 'Code', 'Message', 'Status']].copy()
    df_faults = df_faults.rename(columns={
        'Code': 'code', 'Message': 'message', 'Status': 'status'
    }).reset_index(drop=True)

    logger.info('Fallos técnicos válidos hasta hoy: %d', len(df_faults))
    for fam, cnt in df_faults['family'].value_counts().items():
        logger.info('  %-15s %d', fam, cnt)

    # Cargar fault log existente y hacer merge deduplicado
    try:
        response     = s3_client.get_object(Bucket=BUCKET_NAME, Key=FAULT_LOG_KEY)
        df_existing  = pd.read_csv(io.BytesIO(response['Body'].read()), parse_dates=['timestamp'])
    except ClientError as e:
        if e.response['Error']['Code'] == 'NoSuchKey':
            df_existing = None
        else:
            raise

    if df_existing is not None:
        df_combined = pd.concat([df_existing, df_faults], ignore_index=True)
        df_combined = df_combined.drop_duplicates(
            subset=['timestamp', 'family', 'code'], keep='first'
        ).sort_values('timestamp').reset_index(drop=True)
        n_new = len(df_combined) - len(df_existing)
        logger.info('Fault log actualizado: +%d nuevos registros (%d total).', n_new, len(df_combined))
    else:
        df_combined = df_faults
        logger.info('Fault log creado con %d registros.', len(df_combined))

    csv_buf = io.StringIO()
    df_combined.to_csv(csv_buf, index=False)
    s3_client.put_object(Bucket=BUCKET_NAME, Key=FAULT_LOG_KEY, Body=csv_buf.getvalue())
    logger.info('Fault log guardado en S3: %s', FAULT_LOG_KEY)

    return df_combined


def label_t2_features(df: pd.DataFrame, fault_log: pd.DataFrame,
                       family: str, cfg: dict) -> pd.DataFrame:
    """
    Añade hours_to_fault e is_pre_fault al Feature Store de T2.

    Reglas:
    - Solo fallos con timestamp <= hoy (no usar datos futuros)
    - Floor al intervalo de 10min para alinear con la telemetría
    - Ventana positiva: [fault_ts - lead_hours, fault_ts)
    - Exclusión post-fallo: [fault_ts, fault_ts + POST_FAULT_EXCLUSION_H) → descartadas
    - El resto son negativos (hours_to_fault = lead_hours, is_pre = False)
    """
    lead_hours = cfg['lead_hours']
    hoy        = pd.Timestamp.now()

    # Fallos de esta familia hasta hoy — floor a 10min para alinear con telemetría
    family_faults = fault_log[
        (fault_log['family'] == family) &
        (fault_log['timestamp'] <= hoy)
    ]['timestamp'].copy()
    family_faults = family_faults.dt.floor('10min').sort_values().reset_index(drop=True)

    df = df.copy()
    hours_col  = f'hours_to_{family}'
    target_col = f'is_pre_{family}'

    # Inicializar: todos negativos
    df[hours_col]  = float(lead_hours)
    df[target_col] = False

    # Máscara de exclusión post-fallo (filas a descartar)
    exclude_mask = pd.Series(False, index=df.index)

    for fault_ts in family_faults:
        window_start = fault_ts - pd.Timedelta(hours=lead_hours)
        post_end     = fault_ts + pd.Timedelta(hours=POST_FAULT_EXCLUSION_H)

        # Ventana positiva
        pre_mask = (df['timestamp'] >= window_start) & (df['timestamp'] < fault_ts)
        if pre_mask.any():
            delta_h = (fault_ts - df.loc[pre_mask, 'timestamp']).dt.total_seconds() / 3600
            df.loc[pre_mask, hours_col]  = delta_h.values
            df.loc[pre_mask, target_col] = True

        # Marcar post-fallo para exclusión
        post_mask = (df['timestamp'] >= fault_ts) & (df['timestamp'] < post_end)
        exclude_mask = exclude_mask | post_mask

    # Eliminar filas post-fallo (sensores en transición)
    df = df[~exclude_mask].reset_index(drop=True)

    n_pos = df[target_col].sum()
    n_tot = len(df)
    logger.info('  Etiquetado [%s]: %d positivos / %d filas totales (%.1f%%)',
                family, n_pos, n_tot, 100 * n_pos / n_tot if n_tot > 0 else 0)

    return df


# =============================================================================
# Entrenamiento + calibración
# =============================================================================

def _temporal_split(df: pd.DataFrame) -> tuple:
    """Split temporal 60/20/20 sobre un DataFrame con columna timestamp."""
    df = df.sort_values('timestamp').reset_index(drop=True)
    cutoff_train = df['timestamp'].quantile(TRAIN_RATIO)
    cutoff_val   = df['timestamp'].quantile(VAL_RATIO)
    train = df[df['timestamp'] <  cutoff_train]
    val   = df[(df['timestamp'] >= cutoff_train) & (df['timestamp'] < cutoff_val)]
    test  = df[df['timestamp'] >= cutoff_val].copy()
    return train, val, test


def train_and_evaluate(df: pd.DataFrame, family: str, cfg: dict,
                       source_label: str,
                       df_t2_labeled: pd.DataFrame | None = None,
                       test_t2: pd.DataFrame | None = None) -> dict | None:
    """
    Entrena LGBMRegressor + IsotonicRegression sobre df.
    Devuelve dict con pipeline y metricas, o None si no hay datos suficientes.

    Split temporal 60/20/20:
    - T2_only: split directo sobre df.
    - T1+T2:   split por separado en T1 y T2, luego concatena cada tramo.
      Evita que el gap de 4 anos entre fuentes distorsione el corte temporal.
    
    Si test_t2 se proporciona, se usa directamente como test set (para evaluación
    consistente entre versiones sobre el mismo período de T2).
    """
    lead_hours = cfg['lead_hours']
    alert_h    = cfg['alert_h']
    hours_col  = f'hours_to_{family}'
    target_col = f'is_pre_{family}'

    feat_cols = [c for c in df.columns
                 if c not in ['timestamp', target_col, hours_col]
                 and not c.startswith('is_pre_')
                 and not c.startswith('hours_to_')]

    df = df.sort_values('timestamp').reset_index(drop=True)

    # test siempre es el test_t2 fijo — evaluación consistente entre versiones
    if source_label == 'T1+T2' and df_t2_labeled is not None:
        t1_cutoff = pd.Timestamp('2023-01-01')
        df_t1_src = df[df['timestamp'] < t1_cutoff].copy()
        df_t2_src = df[df['timestamp'] >= t1_cutoff].copy()

        if len(df_t1_src) > 0:
            train_t1, val_t1, _ = _temporal_split(df_t1_src)
        else:
            train_t1 = val_t1 = pd.DataFrame(columns=df.columns)

        if len(df_t2_src) > 0:
            train_t2, val_t2, _ = _temporal_split(df_t2_src)
        else:
            train_t2 = val_t2 = pd.DataFrame(columns=df.columns)

        train = pd.concat([train_t1, train_t2], ignore_index=True)
        val   = pd.concat([val_t1,   val_t2],   ignore_index=True)
        logger.info('  [T1+T2] T1 train/val: %d/%d | T2 train/val: %d/%d',
                    len(train_t1), len(val_t1), len(train_t2), len(val_t2))
    else:
        train, val, _ = _temporal_split(df)
        logger.info('  [T2_only] split temporal para train/val')

    test = test_t2.copy()

    train_pos = train[train[target_col].astype(bool)]
    val_pos   = val[val[target_col].astype(bool)]
    test_pos  = test[test[target_col].astype(bool)]

    logger.info('  [%s] Train pos: %d | Val pos: %d | Test pos: %d',
                source_label, len(train_pos), len(val_pos), len(test_pos))

    if len(train_pos) < 10 or len(val_pos) < 5:
        logger.warning('  [%s] Datos insuficientes para entrenar %s. Saltando.',
                       source_label, family)
        return None

    X_train = train_pos[feat_cols].fillna(0)
    y_train = train_pos[hours_col].values
    X_val   = val_pos[feat_cols].fillna(0)
    y_val   = val_pos[hours_col].values

    # --- LGBMRegressor ---
    model = lgb.LGBMRegressor(**LGBM_PARAMS[family])
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)],
    )

    pred_val_raw = np.clip(model.predict(X_val), 0, lead_hours)

    # --- Calibrador isotónico ---
    calibrator = IsotonicRegression(out_of_bounds='clip', increasing=True)
    calibrator.fit(pred_val_raw, y_val)

    # --- Evaluación sobre test completo (pos + neg) ---
    # Asegurar que test solo tenga columnas que existen en el modelo
    test_feat_cols = [c for c in feat_cols if c in test.columns]
    pred_test_raw = np.clip(model.predict(test[test_feat_cols].fillna(0)), 0, lead_hours)
    pred_test_cal = np.clip(calibrator.predict(pred_test_raw), 0, lead_hours)

    mae_raw = float(mean_absolute_error(
        test_pos[hours_col].values,
        np.clip(model.predict(test_pos[test_feat_cols].fillna(0)), 0, lead_hours)
    )) if len(test_pos) > 0 else None

    mae_cal = float(mean_absolute_error(
        test_pos[hours_col].values,
        np.clip(calibrator.predict(
            np.clip(model.predict(test_pos[test_feat_cols].fillna(0)), 0, lead_hours)
        ), 0, lead_hours)
    )) if len(test_pos) > 0 else None

    # --- Precision / Recall fila a fila ---
    y_true_bin = test[target_col].astype(int).values
    y_pred_bin = (pred_test_cal <= alert_h).astype(int)
    tp = int(((y_pred_bin == 1) & (y_true_bin == 1)).sum())
    fp = int(((y_pred_bin == 1) & (y_true_bin == 0)).sum())
    fn = int(((y_pred_bin == 0) & (y_true_bin == 1)).sum())
    precision = round(tp / (tp + fp), 4) if (tp + fp) > 0 else 0.0
    recall    = round(tp / (tp + fn), 4) if (tp + fn) > 0 else 0.0

    # --- Event Recall ---
    ev = _evaluate_by_event(test, pred_test_cal, family, alert_h, lead_hours)

    # --- F1 entre Event Recall y Precision ---
    er  = ev['event_recall']
    f1_ep = round(2 * er * precision / (er + precision), 4) if (er + precision) > 0 else 0.0

    logger.info('  [%s] MAE bruto: %s | MAE cal: %s | Precision: %.3f | Recall: %.3f | Event Recall: %.3f (%d/%d) | F1(ER,P): %.3f',
                source_label,
                f'{mae_raw:.1f}h' if mae_raw else 'N/A',
                f'{mae_cal:.1f}h' if mae_cal else 'N/A',
                precision, recall, ev['event_recall'],
                ev['events_detected'], ev['events_total'], f1_ep)

    pipeline = {
        'lgbm':         model,
        'calibrator':   calibrator,
        'feature_cols': feat_cols,
        'family':       family,
        'cfg':          cfg,
        'source':       source_label,
    }

    return {
        'pipeline':        pipeline,
        'event_recall':    ev['event_recall'],
        'events_detected': ev['events_detected'],
        'events_total':    ev['events_total'],
        'precision':       precision,
        'recall':          recall,
        'f1_ep':           f1_ep,
        'mae_raw_h':       round(mae_raw, 1) if mae_raw else None,
        'mae_cal_h':       round(mae_cal, 1) if mae_cal else None,
        'source':          source_label,
        'n_estimators':    int(model.best_iteration_) if hasattr(model, 'best_iteration_') else 1000,
        'test_t2_start':   str(test['timestamp'].min().date()),
        'test_t2_end':     str(test['timestamp'].max().date()),
        'test_t2_rows':    len(test),  # ← Añadido para que el handler lo use
    }


def _evaluate_by_event(test_df: pd.DataFrame, y_pred_h: np.ndarray,
                       family: str, alert_h: float, lead_hours: float) -> dict:
    """
    Event Recall: fracción de fallos reales que recibieron al menos
    una alerta (pred_h <= alert_h) dentro de su ventana de lead time.
    Idéntico al notebook 07_calibration.
    """
    target_col = f'is_pre_{family}'
    df_e = test_df[['timestamp', target_col]].copy()
    df_e['pred_h'] = y_pred_h
    df_e = df_e.sort_values('timestamp').reset_index(drop=True)

    is_pre = df_e[target_col].astype(bool)
    fault_starts = df_e['timestamp'][
        is_pre & ~is_pre.shift(1, fill_value=False)
    ].tolist()

    detected = 0
    for fs in fault_starts:
        mask     = (df_e['timestamp'] >= fs) & \
                   (df_e['timestamp'] < fs + pd.Timedelta(hours=lead_hours))
        min_pred = float(df_e.loc[mask, 'pred_h'].min()) if mask.any() else float(lead_hours)
        if min_pred <= alert_h:
            detected += 1

    total = len(fault_starts)
    return {
        'event_recall':    round(detected / total, 4) if total > 0 else 0.0,
        'events_detected': detected,
        'events_total':    total,
    }


# =============================================================================
# Handler Lambda
# =============================================================================

def handler(event, context):
    logger.info('=' * 60)
    logger.info('REENTRENAMIENTO MENSUAL — Turbina %d', TURBINE_ID)
    logger.info('=' * 60)

    s3_client = boto3.client('s3')
    hoy       = pd.Timestamp.now()
    logger.info('Fecha de reentrenamiento: %s', hoy.date())

    # -------------------------------------------------------------------------
    # PASO 0: Actualizar fault log desde CSV de eventos SCADA en S3
    # Filtra por timestamp <= hoy antes de procesar (simulación hasta 2030)
    # -------------------------------------------------------------------------
    logger.info('PASO 0: Actualizando fault log desde eventos SCADA...')
    fault_log = update_fault_log(s3_client, hoy)

    if len(fault_log) == 0:
        logger.warning('Fault log vacío hasta hoy. Sin fallos reales de T2 para etiquetar.')
        return {'statusCode': 200, 'body': 'Sin fallos de T2. Reentrenamiento omitido.'}

    # -------------------------------------------------------------------------
    # PASO 1: Fault log ya cargado y actualizado en PASO 0
    # -------------------------------------------------------------------------
    logger.info('Fault log listo: %d fallos hasta %s.', len(fault_log), hoy.date())

    # -------------------------------------------------------------------------
    # PASO 2: Cargar Feature Stores de T1 (ya etiquetados desde notebooks)
    # -------------------------------------------------------------------------
    t1_stores = {}
    for family in FAMILIES:
        key = f'models/t1_features_{family}.parquet'
        df  = _load_parquet(s3_client, key)
        if df is not None:
            t1_stores[family] = df
            logger.info('T1 Feature Store [%s]: %d filas.', family, len(df))
        else:
            logger.warning('T1 Feature Store no encontrado: %s', key)

    # -------------------------------------------------------------------------
    # PASO 3: Por familia — etiquetar T2, entrenar dos versiones, desplegar ganadora
    # -------------------------------------------------------------------------
    retrain_results = {
        'retrained_at': hoy.isoformat(),
        'families':     {},
    }

    for family, cfg in FAMILIES.items():
        t0 = time.time()
        logger.info('')
        logger.info('=' * 55)
        logger.info('FAMILIA: %s', family.upper())
        logger.info('=' * 55)

        # Cargar Feature Store de T2
        t2_key = f'models/t2_features_{family}.parquet'
        df_t2  = _load_parquet(s3_client, t2_key)
        if df_t2 is None or len(df_t2) == 0:
            logger.warning('Feature Store de T2 vacío para %s. Saltando.', family)
            continue

        # Etiquetar T2 con fallos reales (floor a 10min)
        df_t2_labeled = label_t2_features(df_t2, fault_log, family, cfg)

        n_pos = df_t2_labeled[f'is_pre_{family}'].sum()
        if n_pos == 0:
            logger.warning('Sin positivos etiquetados en T2 para %s. Saltando.', family)
            continue

        # test_t2: 20% temporal final de T2 etiquetado.
        # Se calcula una sola vez y se pasa a ambas versiones para evaluacion justa.
        _, _, test_t2 = _temporal_split(df_t2_labeled)
        n_test_t2_pos = int(test_t2[f'is_pre_{family}'].sum())
        logger.info('Test set T2: %d filas, %d positivos.', len(test_t2), n_test_t2_pos)

        # --- Version A: Solo T2 ---
        logger.info('Entrenando version A (solo T2)...')
        result_t2_only = train_and_evaluate(df_t2_labeled, family, cfg, 'T2_only',
                                            test_t2=test_t2)

        # --- Version B: T1 + T2 ---
        result_t1_t2 = None
        if family in t1_stores:
            logger.info('Entrenando version B (T1 + T2)...')
            df_t1 = t1_stores[family]

            feat_cols_t2 = [c for c in df_t2_labeled.columns
                            if c not in ['timestamp', f'is_pre_{family}', f'hours_to_{family}']
                            and not c.startswith('is_pre_')
                            and not c.startswith('hours_to_')]
            feat_cols_t1 = [c for c in df_t1.columns
                            if c not in ['timestamp', f'is_pre_{family}', f'hours_to_{family}']
                            and not c.startswith('is_pre_')
                            and not c.startswith('hours_to_')]
            common_cols = list(set(feat_cols_t1) & set(feat_cols_t2))

            keep_cols = ['timestamp', f'is_pre_{family}', f'hours_to_{family}'] + common_cols
            df_t1_trim = df_t1[[c for c in keep_cols if c in df_t1.columns]].copy()
            df_t2_trim = df_t2_labeled[[c for c in keep_cols if c in df_t2_labeled.columns]].copy()

            df_combined = pd.concat([df_t1_trim, df_t2_trim], ignore_index=True)
            df_combined = df_combined.sort_values('timestamp').reset_index(drop=True)

            # test_t2 filtrado a columnas comunes para evaluacion consistente
            test_t2_trim = test_t2[[c for c in test_t2.columns if c in keep_cols]].copy()

            result_t1_t2 = train_and_evaluate(df_combined, family, cfg, 'T1+T2',
                                              test_t2=test_t2_trim,
                                              df_t2_labeled=df_t2_labeled)
        else:
            logger.warning('T1 Feature Store no disponible para %s. Solo se entrena version T2.', family)

        # --- Seleccionar ganador por Event Recall en test set de T2 ---
        #
        # Reglas de selección:
        # 1. Si hay menos de MIN_TEST_EVENTS_FOR_COMPARISON eventos en el test set
        #    de T2, la comparación no es estadísticamente válida → forzar T1+T2.
        # 2. Si hay empate en Event Recall → T1+T2 gana (más datos = más robusto).
        # 3. Si solo hay una versión disponible → esa gana.
        candidates = [r for r in [result_t2_only, result_t1_t2] if r is not None]
        if not candidates:
            logger.warning('Ninguna versión entrenada para %s.', family)
            continue

        # Selección del ganador por F1(Event Recall, Precision) sobre test_t2.
        # Ambas versiones se evaluaron sobre el mismo test_t2 → comparación justa.
        # Empate → T1+T2 por más datos de entrenamiento.
        if len(candidates) == 1:
            winner = candidates[0]
            selection_reason = 'única versión disponible'
        else:
            best_f1 = max(r['f1_ep'] for r in candidates)
            winners_by_f1 = [r for r in candidates if r['f1_ep'] == best_f1]
            if len(winners_by_f1) > 1:
                winner = result_t1_t2 if result_t1_t2 is not None else result_t2_only
                selection_reason = f'empate F1(ER,P)={best_f1:.3f} → T1+T2 por más datos'
            else:
                winner = winners_by_f1[0]
                selection_reason = f'mejor F1(ER,P)={best_f1:.3f} (ER={winner["event_recall"]:.3f}, P={winner["precision"]:.3f})'

        loser = [r for r in candidates if r is not winner]
        logger.info('')
        logger.info('  GANADOR [%s]: Event Recall = %.3f  (%s)',
                    winner['source'], winner['event_recall'], selection_reason)
        if loser:
            logger.info('  Perdedor [%s]: Event Recall = %.3f',
                        loser[0]['source'], loser[0]['event_recall'])

        # --- Desplegar modelo ganador en S3 ---
        model_key = f'models/t2_model_{family}.pkl'
        _save_pickle(s3_client, winner['pipeline'], model_key)
        logger.info('  Modelo desplegado: s3://%s/%s', BUCKET_NAME, model_key)

        elapsed = round(time.time() - t0, 1)
        retrain_results['families'][family] = {
            'winner':            winner['source'],
            'selection_reason':  selection_reason,
            'event_recall':      winner['event_recall'],
            'events_detected':   winner['events_detected'],
            'events_total':      winner['events_total'],
            'precision':         winner['precision'],
            'recall':            winner['recall'],
            'f1_ep':             winner['f1_ep'],
            'mae_cal_h':         winner['mae_cal_h'],
            'n_estimators':      winner['n_estimators'],
            'test_t2_start':     winner['test_t2_start'],
            'test_t2_end':       winner['test_t2_end'],
            'test_t2_rows':      winner['test_t2_rows'],
            'candidates': {r['source']: {
                'event_recall': r['event_recall'],
                'precision':    r['precision'],
                'recall':       r['recall'],
                'f1_ep':        r['f1_ep'],
                'mae_cal_h':    r['mae_cal_h'],
            } for r in candidates},
            'elapsed_s': elapsed,
        }

    # -------------------------------------------------------------------------
    # PASO 4: Guardar resultados del reentrenamiento en S3
    #
    # Dos formatos:
    #   A) t2_retrain_results.json  — último reentrenamiento completo (para dashboard)
    #   B) t2_retrain_log.csv       — histórico acumulativo de todos los reentrenamientos
    # -------------------------------------------------------------------------
    results_key = 'models/t2_retrain_results.json'
    _save_json(s3_client, retrain_results, results_key)
    logger.info('Resultados JSON guardados en s3://%s/%s', BUCKET_NAME, results_key)

    # Construir filas para el log CSV acumulativo
    log_rows = []
    for fam, res in retrain_results['families'].items():
        row_base = {
            'retrained_at':     retrain_results['retrained_at'],
            'family':           fam,
            'winner':           res['winner'],
            'selection_reason': res['selection_reason'],
            'event_recall':     res['event_recall'],
            'events_detected':  res['events_detected'],
            'events_total':     res['events_total'],
            'precision':        res['precision'],
            'recall':           res['recall'],
            'f1_ep':            res['f1_ep'],
            'mae_cal_h':        res['mae_cal_h'],
            'n_estimators':     res['n_estimators'],
            'test_t2_start':    res['test_t2_start'],
            'test_t2_end':      res['test_t2_end'],
            'test_t2_rows':     res['test_t2_rows'],
            'elapsed_s':        res['elapsed_s'],
        }
        # Añadir métricas de cada candidato como columnas planas
        for src, metrics in res['candidates'].items():
            safe = src.replace('+', '_')
            row_base[f'er_{safe}']        = metrics['event_recall']
            row_base[f'precision_{safe}'] = metrics['precision']
            row_base[f'recall_{safe}']    = metrics['recall']
            row_base[f'f1_ep_{safe}']     = metrics['f1_ep']
            row_base[f'mae_{safe}']       = metrics['mae_cal_h']
        log_rows.append(row_base)

    df_new_log = pd.DataFrame(log_rows)

    # Append al log acumulativo
    log_key = 'models/t2_retrain_log.csv'
    try:
        response = s3_client.get_object(Bucket=BUCKET_NAME, Key=log_key)
        df_existing_log = pd.read_csv(io.BytesIO(response['Body'].read()))
        df_retrain_log  = pd.concat([df_existing_log, df_new_log], ignore_index=True)
        df_retrain_log  = df_retrain_log.drop_duplicates(
            subset=['retrained_at', 'family'], keep='last'
        ).reset_index(drop=True)
        logger.info('Log acumulativo actualizado: %d registros totales.', len(df_retrain_log))
    except ClientError as e:
        if e.response['Error']['Code'] == 'NoSuchKey':
            df_retrain_log = df_new_log
            logger.info('Log acumulativo creado.')
        else:
            raise

    csv_buf = io.StringIO()
    df_retrain_log.to_csv(csv_buf, index=False)
    s3_client.put_object(Bucket=BUCKET_NAME, Key=log_key, Body=csv_buf.getvalue())
    logger.info('Log acumulativo guardado en s3://%s/%s', BUCKET_NAME, log_key)

    logger.info('')
    logger.info('=' * 60)
    logger.info('REENTRENAMIENTO COMPLETADO')
    logger.info('=' * 60)
    for family, res in retrain_results['families'].items():
        logger.info('  %-15s ganador=%-8s Event Recall=%.3f (%d/%d)',
                    family, res['winner'], res['event_recall'],
                    res['events_detected'], res['events_total'])

    return {
        'statusCode': 200,
        'body': json.dumps(retrain_results, default=str),
    }


if __name__ == '__main__':
    print('🚀 Reentrenamiento mensual local...')
    handler({}, None)
