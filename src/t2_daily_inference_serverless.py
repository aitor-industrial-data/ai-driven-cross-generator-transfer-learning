import os
import json
import pickle
import logging
import io
import sys
import boto3
import pandas as pd
import numpy as np
from botocore.exceptions import ClientError

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src', 'shared'))
try:
    from feature_builder import (
        FAMILY_SENSORS, add_domain_features,
        make_rolling_features, add_temporal_context
    )
except ImportError:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'shared'))
    from feature_builder import (
        FAMILY_SENSORS, add_domain_features,
        make_rolling_features, add_temporal_context
    )

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

BUCKET_NAME = os.environ.get('AWS_S3_BUCKET', 'ai-driven-cross-generator-transfer-learning')
TURBINE_ID  = 2

FAMILIES = {
    'yaw_cable':   {'alert_h': 48,  'lead_hours': 83},
    'generator':   {'alert_h': 72,  'lead_hours': 127},
    'brake_hydro': {'alert_h': 72,  'lead_hours': 130},
    'pitch_bat':   {'alert_h': 168, 'lead_hours': 295},
}

# 7 días a pasos de 10 minutos
STEPS_7D = 1008

# Cold start: horas asignadas a hours_since_last_fault cuando T2 no tiene fallos propios.
# Se usa el máximo razonable del dominio de entrenamiento de T1 para no extrapolar.
COLD_START_HOURS = 9999.0

# Ruta en S3 de cada Feature Store por familia
# Estructura idéntica a T1: t2_features_{family}.parquet
# Columnas: timestamp | rolling features | hours_since_last_{family} | hours_since_last_{family}_log
# (sin targets hours_to_fault / is_pre_fault — T2 está en producción, no en entrenamiento)
def _feature_store_key(family: str) -> str:
    return f'models/t2_features_{family}.parquet'


def _load_parquet_from_s3(s3_client, key: str) -> pd.DataFrame | None:
    """Descarga un Parquet de S3. Devuelve None si no existe."""
    try:
        response = s3_client.get_object(Bucket=BUCKET_NAME, Key=key)
        return pd.read_parquet(io.BytesIO(response['Body'].read()))
    except ClientError as e:
        if e.response['Error']['Code'] == 'NoSuchKey':
            return None
        raise


def _get_last_stored_ts(df_store: pd.DataFrame | None) -> pd.Timestamp | None:
    """Devuelve el último timestamp guardado en el Feature Store, o None si está vacío."""
    if df_store is None or len(df_store) == 0:
        return None
    return df_store['timestamp'].max()


def _normalize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fuerza tipos consistentes para serialización PyArrow.
    El concat de múltiples DataFrames puede dejar columnas object cuando
    hay NaN mezclados con floats.
    """
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    for col in df.columns:
        if col == 'timestamp':
            continue
        dtype = str(df[col].dtype)
        if dtype == 'object':
            df[col] = pd.to_numeric(df[col], errors='coerce').astype('float32')
        elif dtype == 'float64':
            df[col] = df[col].astype('float32')
    return df


def _save_parquet_to_s3(s3_client, df: pd.DataFrame, key: str) -> None:
    """Serializa un DataFrame como Parquet y lo sube a S3."""
    buf = io.BytesIO()
    df.to_parquet(buf, index=False, engine='pyarrow')
    s3_client.put_object(Bucket=BUCKET_NAME, Key=key, Body=buf.getvalue())


def handler(event, context):
    """Punto de entrada ejecutado por AWS Lambda."""
    logger.info('=' * 60)
    logger.info('INFERENCIA DIARIA SERVERLESS — Turbina %d', TURBINE_ID)
    logger.info('=' * 60)

    s3_client = boto3.client('s3')
    ahora = pd.Timestamp.now()
    logger.info('Timestamp de ejecución: %s', ahora)

    # -------------------------------------------------------------------------
    # PASO 1: Cargar telemetría Bronze desde S3
    # -------------------------------------------------------------------------
    bronze_key = f'bronze/turbine_{TURBINE_ID}_telemetry_clean.parquet/'
    logger.info('Descargando telemetría Bronze: s3://%s/%s', BUCKET_NAME, bronze_key)

    try:
        df_bronze = pd.read_parquet(f's3://{BUCKET_NAME}/{bronze_key}')
        df_bronze = df_bronze.sort_values('timestamp').reset_index(drop=True)
        logger.info('Bronze cargado: %d filas totales.', len(df_bronze))
    except Exception as e:
        logger.error('Error crítico leyendo Bronze: %s', str(e))
        return {'statusCode': 500, 'body': f'Error cargando telemetría: {str(e)}'}

    df_bronze = df_bronze[df_bronze['timestamp'] <= ahora].copy()
    if len(df_bronze) == 0:
        logger.error('Bronze sin registros hasta %s.', ahora)
        return {'statusCode': 400, 'body': 'Sin registros en Bronze para la fecha actual.'}

    # -------------------------------------------------------------------------
    # PASO 2: Cargar Baseline de T2 desde S3
    # -------------------------------------------------------------------------
    baseline_key = f'models/turbine_{TURBINE_ID}_baseline.json'
    try:
        response = s3_client.get_object(Bucket=BUCKET_NAME, Key=baseline_key)
        bl = json.loads(response['Body'].read().decode('utf-8'))
        baseline_mean = bl['mean']
        baseline_p90  = bl['p90']
        logger.info('Baseline de T2 cargado correctamente.')
    except Exception as e:
        logger.warning('Baseline no encontrado (%s). Usando diccionarios vacíos.', str(e))
        baseline_mean = {}
        baseline_p90  = {}

    # -------------------------------------------------------------------------
    # PASO 3: Cargar Log de Fallos de T2 desde S3
    # -------------------------------------------------------------------------
    fault_log_key = f'models/turbine_{TURBINE_ID}_fault_log.csv'
    try:
        response = s3_client.get_object(Bucket=BUCKET_NAME, Key=fault_log_key)
        fault_log = pd.read_csv(io.BytesIO(response['Body'].read()), parse_dates=['timestamp'])
        # El fault log ya está filtrado por fecha — lo actualiza el reentrenamiento mensual (PASO 0)
        logger.info('Fault log de T2 cargado: %d registros.', len(fault_log))
    except ClientError as e:
        if e.response['Error']['Code'] == 'NoSuchKey':
            logger.info('Fault log no encontrado. T2 en cold start (sin fallos propios registrados).')
            fault_log = pd.DataFrame(columns=['timestamp', 'family'])
        else:
            raise

    # -------------------------------------------------------------------------
    # PASO 4: Pre-calcular ventana de cálculo con domain features
    #
    # Se hace una sola vez para todas las familias, ya que add_domain_features
    # es común. Cada familia usará esta misma ventana para sus rolling features.
    # -------------------------------------------------------------------------

    # Determinar el último timestamp ya almacenado en cualquier Feature Store.
    # Usamos el mínimo entre familias para garantizar que todas están sincronizadas.
    last_stored_ts_per_family = {}
    for family in FAMILIES:
        df_store = _load_parquet_from_s3(s3_client, _feature_store_key(family))
        last_stored_ts_per_family[family] = _get_last_stored_ts(df_store)
        if last_stored_ts_per_family[family] is not None:
            logger.info('Feature Store [%s]: último timestamp = %s (%d filas)',
                        family, last_stored_ts_per_family[family], len(df_store))
        else:
            logger.info('Feature Store [%s]: no existe todavía.', family)

    # El timestamp de corte para nuevas filas es el mínimo entre familias
    # (si una familia va rezagada, recalculamos desde ahí para todas)
    stored_ts_values = [ts for ts in last_stored_ts_per_family.values() if ts is not None]
    global_last_stored_ts = min(stored_ts_values) if stored_ts_values else None

    if global_last_stored_ts is not None:
        df_new_rows = df_bronze[df_bronze['timestamp'] > global_last_stored_ts].copy()
    else:
        df_new_rows = df_bronze.copy()

    logger.info('Filas nuevas a procesar: %d', len(df_new_rows))

    if len(df_new_rows) > 0:
        primer_ts_new = df_new_rows['timestamp'].min()

        # Contexto rolling: STEPS_7D filas previas al primer timestamp nuevo
        df_contexto_previo = df_bronze[df_bronze['timestamp'] < primer_ts_new].tail(STEPS_7D)
        df_calc_window     = pd.concat([df_contexto_previo, df_new_rows], ignore_index=True)
        df_calc_window     = add_domain_features(df_calc_window)

        logger.info('Ventana de cálculo: %d filas (%s → %s)',
                    len(df_calc_window),
                    df_calc_window['timestamp'].min().date(),
                    df_calc_window['timestamp'].max().date())
    else:
        df_calc_window = None
        primer_ts_new  = None

    # -------------------------------------------------------------------------
    # PASO 5: Por familia — calcular features, actualizar Feature Store
    #
    # Estructura de cada t2_features_{family}.parquet (idéntica a T1):
    #   timestamp | {sensor}__mean_1h | ... | hours_since_last_{family} | hours_since_last_{family}_log
    #
    # No incluye targets (hours_to_fault, is_pre_fault) porque T2 está en producción.
    # El notebook de reentrenamiento mensual une T1 (con targets) + T2 (sin targets)
    # usando transfer learning o fine-tuning.
    # -------------------------------------------------------------------------
    family_stores = {}  # family → df_store actualizado (para inferencia en PASO 6)

    for family, cfg in FAMILIES.items():
        logger.info('--- Feature Store [%s] ---', family)

        store_key = _feature_store_key(family)
        df_store  = _load_parquet_from_s3(s3_client, store_key)

        if df_calc_window is not None:
            fault_times = fault_log[fault_log['family'] == family]['timestamp'].tolist()

            # Rolling features para esta familia
            feats_rolling   = make_rolling_features(
                df_calc_window, FAMILY_SENSORS[family], baseline_mean, baseline_p90
            )
            df_family_feats = pd.concat([df_calc_window[['timestamp']], feats_rolling], axis=1)
            df_family_feats = add_temporal_context(df_family_feats, family, fault_times)

            # Cold start: T2 sin fallos propios → COLD_START_HOURS
            if not fault_times:
                df_family_feats[f'hours_since_last_{family}']     = COLD_START_HOURS
                df_family_feats[f'hours_since_last_{family}_log'] = float(np.log1p(COLD_START_HOURS))

            # Filtrar: solo filas nuevas (las del contexto previo sirvieron solo para rolling)
            df_new_family = df_family_feats[
                df_family_feats['timestamp'] >= primer_ts_new
            ].reset_index(drop=True)

            # Append al Feature Store de esta familia
            if df_store is not None:
                df_updated = pd.concat([df_store, df_new_family], ignore_index=True)
                df_updated = df_updated.drop_duplicates(subset=['timestamp'], keep='last').reset_index(drop=True)
            else:
                df_updated = df_new_family

            df_updated = _normalize_dtypes(df_updated)
            _save_parquet_to_s3(s3_client, df_updated, store_key)
            logger.info('  Feature Store [%s] actualizado: %d filas totales.', family, len(df_updated))
            family_stores[family] = df_updated
        else:
            # Sin filas nuevas: usar el store existente para inferencia
            family_stores[family] = df_store
            if df_store is not None:
                logger.info('  Feature Store [%s]: sin cambios (%d filas).', family, len(df_store))
            else:
                logger.warning('  Feature Store [%s]: vacío y sin datos nuevos.', family)

    # -------------------------------------------------------------------------
    # PASO 6: Inferencia — cada familia usa las últimas STEPS_7D de su Feature Store
    # -------------------------------------------------------------------------
    ahora_ts  = pd.Timestamp.now()
    hoy       = ahora_ts.normalize()
    pred_log_key  = 'models/t2_predictions_log.csv'
    results_today = []

    for family, cfg in FAMILIES.items():
        logger.info('--- Inferencia [%s] ---', family)

        df_store = family_stores.get(family)
        if df_store is None or len(df_store) == 0:
            logger.warning('  Feature Store vacío para %s. Saltando.', family)
            continue

        # Buscar primero modelo propio de T2 (reentrenado), si no existe usar T1 como fallback
        model_key = f'models/t2_model_{family}.pkl'
        try:
            response = s3_client.get_object(Bucket=BUCKET_NAME, Key=model_key)
            pipeline = pickle.loads(response['Body'].read())
            logger.info('  Usando modelo T2 propio: %s', model_key)
        except ClientError as e:
            if e.response['Error']['Code'] != 'NoSuchKey':
                raise
            model_key = f'models/t1_model_{family}.pkl'
            logger.info('  Modelo T2 no existe aún, usando fallback T1: %s', model_key)
            try:
                response = s3_client.get_object(Bucket=BUCKET_NAME, Key=model_key)
                pipeline = pickle.loads(response['Body'].read())
            except ClientError as e2:
                if e2.response['Error']['Code'] == 'NoSuchKey':
                    logger.warning('  Modelo T1 tampoco encontrado: %s', model_key)
                    continue
                raise

        lgbm         = pipeline['lgbm']
        calibrator   = pipeline['calibrator']
        feature_cols = pipeline['feature_cols']

        df_window = df_store.tail(STEPS_7D).reset_index(drop=True)
        ultimo_ts = df_window['timestamp'].max()

        # Columnas ausentes: pueden ocurrir en las primeras ejecuciones
        # cuando el Feature Store aún no tiene suficiente historial
        for col in feature_cols:
            if col not in df_window.columns:
                logger.warning('  Columna ausente, rellenando con 0: %s', col)
                df_window[col] = 0.0

        X_hoy = df_window[feature_cols].fillna(0).iloc[[-1]]

        raw_pred = float(np.clip(lgbm.predict(X_hoy)[0], 0, cfg['lead_hours']))
        cal_pred = float(np.clip(calibrator.predict([raw_pred])[0], 0, cfg['lead_hours']))
        is_alert = cal_pred <= cfg['alert_h']

        logger.info('  Predicción calibrada: %.1fh | Alerta: %s',
                    cal_pred, '🚨 SÍ' if is_alert else '✅ NO')

        results_today.append({
            'date':              str(hoy.date()),
            'last_data_ts':      str(ultimo_ts),
            'turbine':           TURBINE_ID,
            'family':            family,
            'pred_h':            round(cal_pred, 1),
            'alert':             is_alert,
            'alert_threshold_h': cfg['alert_h'],
        })

    if not results_today:
        return {'statusCode': 200, 'body': 'No se ejecutaron predicciones (modelos no encontrados).'}

    # -------------------------------------------------------------------------
    # PASO 7: Actualizar Log de Predicciones en S3 (append)
    #
    # Deduplicación por last_data_ts + family: idempotente si se ejecuta
    # varias veces con los mismos datos. Acumula si hay datos nuevos.
    # NUNCA sobreescribe el historial si hay un error leyendo el log existente.
    # -------------------------------------------------------------------------
    df_results      = pd.DataFrame(results_today)
    df_existing_log = None

    try:
        response = s3_client.get_object(Bucket=BUCKET_NAME, Key=pred_log_key)
        df_existing_log = pd.read_csv(io.BytesIO(response['Body'].read()))
        logger.info('Log existente cargado: %d registros.', len(df_existing_log))
    except ClientError as e:
        if e.response['Error']['Code'] == 'NoSuchKey':
            logger.info('Log de predicciones no existe aún. Se creará.')
        else:
            logger.error('Error S3 leyendo log de predicciones: %s', str(e))
            raise
    except Exception as e:
        logger.error('Error parseando log de predicciones existente: %s', str(e))
        raise

    if df_existing_log is not None:
        df_updated_log = pd.concat([df_existing_log, df_results], ignore_index=True)
        df_updated_log = df_updated_log.drop_duplicates(
            subset=['last_data_ts', 'family'], keep='last'
        ).reset_index(drop=True)
    else:
        df_updated_log = df_results

    csv_buffer = io.StringIO()
    df_updated_log.to_csv(csv_buffer, index=False)
    s3_client.put_object(Bucket=BUCKET_NAME, Key=pred_log_key, Body=csv_buffer.getvalue())
    logger.info('Log de predicciones guardado: %d registros totales.', len(df_updated_log))

    logger.info('=' * 60)
    logger.info('Pipeline completado. %d predicciones guardadas.', len(results_today))
    logger.info('=' * 60)

    return {
        'statusCode': 200,
        'body': json.dumps(results_today)
    }


if __name__ == '__main__':
    print('🚀 Prueba local del Handler de AWS Lambda...')
    handler({}, None)