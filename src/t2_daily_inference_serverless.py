import os
import json
import pickle
import logging
import io
import sys
import boto3
import pandas as pd
import numpy as np

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

# Valor de cold start para hours_since_last_fault cuando T2 no tiene fallos propios.
# 9999 indica al modelo "lleva mucho tiempo sin fallar / sin historial conocido".
COLD_START_HOURS = 9999.0


def _load_feature_store(s3_client: boto3.client, key: str) -> pd.DataFrame | None:
    """Descarga el Feature Store de S3. Devuelve None si no existe todavía."""
    try:
        response = s3_client.get_object(Bucket=BUCKET_NAME, Key=key)
        df = pd.read_csv(io.BytesIO(response['Body'].read()), parse_dates=['timestamp'])
        return df
    except s3_client.exceptions.NoSuchKey:
        return None


def _get_new_telemetry_rows(df_bronze: pd.DataFrame, df_store: pd.DataFrame | None) -> pd.DataFrame:
    """
    Devuelve las filas de df_bronze que aún no están en el Feature Store.
    Si el Feature Store no existe, devuelve todo df_bronze.
    """
    if df_store is None or len(df_store) == 0:
        return df_bronze.copy()
    last_stored_ts = df_store['timestamp'].max()
    new_rows = df_bronze[df_bronze['timestamp'] > last_stored_ts].copy()
    return new_rows


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

    # Solo registros hasta ahora (producción real)
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
        logger.info('Fault log de T2 cargado: %d registros.', len(fault_log))
    except s3_client.exceptions.NoSuchKey:
        logger.info('Fault log no encontrado. T2 en cold start (sin fallos propios registrados).')
        fault_log = pd.DataFrame(columns=['timestamp', 'family'])

    # -------------------------------------------------------------------------
    # PASO 4: Cargar Feature Store existente e identificar filas nuevas
    # -------------------------------------------------------------------------
    features_log_key = f'models/turbine_{TURBINE_ID}_features_history.parquet'
    df_store = _load_feature_store(s3_client, features_log_key)

    if df_store is not None:
        logger.info('Feature Store cargado: %d filas existentes.', len(df_store))
    else:
        logger.info('Feature Store no existe todavía. Se creará en esta ejecución.')

    df_new_rows = _get_new_telemetry_rows(df_bronze, df_store)
    logger.info('Filas nuevas a procesar e incorporar al Feature Store: %d', len(df_new_rows))

    if len(df_new_rows) == 0:
        logger.info('No hay telemetría nueva desde la última ejecución. Nada que añadir al Feature Store.')

    # -------------------------------------------------------------------------
    # PASO 5: Calcular features para las filas nuevas y actualizar Feature Store
    #
    # Para calcular features rolling correctamente, necesitamos contexto previo.
    # Usamos una ventana extendida: las últimas STEPS_7D filas del Bronze como
    # contexto de cálculo, pero solo añadimos al Feature Store las filas nuevas.
    # -------------------------------------------------------------------------
    if len(df_new_rows) > 0:
        # Contexto de cálculo: últimas STEPS_7D filas del Bronze (incluye filas nuevas)
        ultimo_ts_bronze = df_bronze['timestamp'].max()
        primer_ts_new    = df_new_rows['timestamp'].min()

        # Tomamos las STEPS_7D filas anteriores al primer timestamp nuevo como contexto,
        # más todas las filas nuevas
        df_contexto_previo = df_bronze[df_bronze['timestamp'] < primer_ts_new].tail(STEPS_7D)
        df_calc_window     = pd.concat([df_contexto_previo, df_new_rows], ignore_index=True)
        df_calc_window     = add_domain_features(df_calc_window)

        logger.info('Ventana de cálculo: %d filas (%s → %s)',
                    len(df_calc_window),
                    df_calc_window['timestamp'].min().date(),
                    df_calc_window['timestamp'].max().date())

        # Calculamos features rolling para todas las familias sobre la ventana completa
        # Las features de cada familia tienen prefijo por sensor, sin colisión entre familias
        all_feats_parts = [df_calc_window[['timestamp']].copy()]

        for family, cfg in FAMILIES.items():
            fault_times = fault_log[fault_log['family'] == family]['timestamp'].tolist()
            # Cold start: T2 sin fallos propios → 9999h (no usar fallos de T1)
            # add_temporal_context con lista vacía ya asigna 8760; lo sobreescribimos con 9999
            feats_rolling = make_rolling_features(
                df_calc_window, FAMILY_SENSORS[family], baseline_mean, baseline_p90
            )
            df_family_feats = pd.concat([df_calc_window[['timestamp']], feats_rolling], axis=1)
            df_family_feats = add_temporal_context(df_family_feats, family, fault_times)

            # Sobreescribir cold start con 9999 cuando no hay fallos propios de T2
            if not fault_times:
                df_family_feats[f'hours_since_last_{family}']     = COLD_START_HOURS
                df_family_feats[f'hours_since_last_{family}_log'] = float(np.log1p(COLD_START_HOURS))

            # Adjuntamos las columnas de esta familia (sin timestamp duplicado)
            cols_family = [c for c in df_family_feats.columns if c != 'timestamp']
            all_feats_parts.append(df_family_feats[cols_family].copy())

        # DataFrame completo con todas las features de todas las familias
        df_all_feats = pd.concat(all_feats_parts, axis=1)

        # Añadimos columnas de dominio físico calculadas en add_domain_features
        domain_cols = [c for c in df_calc_window.columns
                       if c not in df_bronze.columns or c == 'timestamp']
        for col in domain_cols:
            if col != 'timestamp' and col not in df_all_feats.columns:
                df_all_feats[col] = df_calc_window[col].values

        # Filtrar: solo guardamos las filas que corresponden a timestamps nuevos
        df_new_features = df_all_feats[
            df_all_feats['timestamp'] >= primer_ts_new
        ].reset_index(drop=True)

        logger.info('Features calculadas para %d filas nuevas.', len(df_new_features))

        # Append al Feature Store
        if df_store is not None:
            df_updated_store = pd.concat([df_store, df_new_features], ignore_index=True)
        else:
            df_updated_store = df_new_features

        # Guardar Feature Store actualizado en S3 (Parquet por eficiencia)
        parquet_buffer = io.BytesIO()
        df_updated_store.to_parquet(parquet_buffer, index=False)
        s3_client.put_object(
            Bucket=BUCKET_NAME,
            Key=features_log_key,
            Body=parquet_buffer.getvalue()
        )
        logger.info('Feature Store actualizado en S3: %d filas totales.', len(df_updated_store))
    else:
        # No hay filas nuevas pero sí tenemos el store para inferencia
        df_updated_store = df_store

    # -------------------------------------------------------------------------
    # PASO 6: Inferencia — usamos las últimas STEPS_7D filas del Feature Store
    # -------------------------------------------------------------------------
    if df_updated_store is None or len(df_updated_store) == 0:
        logger.error('Feature Store vacío. No es posible realizar inferencia.')
        return {'statusCode': 500, 'body': 'Feature Store vacío.'}

    df_inference_window = df_updated_store.tail(STEPS_7D).reset_index(drop=True)
    ultimo_ts = df_inference_window['timestamp'].max()
    primer_ts = df_inference_window['timestamp'].min()
    hoy       = pd.Timestamp.now().normalize()

    logger.info('Ventana de inferencia: %s → %s (%d filas)',
                primer_ts.date(), ultimo_ts.date(), len(df_inference_window))

    pred_log_key  = 'models/t2_predictions_log.csv'
    results_today = []

    for family, cfg in FAMILIES.items():
        logger.info('--- Subsistema: %s ---', family)

        model_key = f'models/t1_model_{family}.pkl'
        try:
            response = s3_client.get_object(Bucket=BUCKET_NAME, Key=model_key)
            pipeline = pickle.loads(response['Body'].read())
        except s3_client.exceptions.NoSuchKey:
            logger.warning('Modelo no encontrado en S3: %s', model_key)
            continue

        lgbm         = pipeline['lgbm']
        calibrator   = pipeline['calibrator']
        feature_cols = pipeline['feature_cols']

        # Rellenar columnas ausentes con 0 (por si feature_cols tiene algo no calculado)
        for col in feature_cols:
            if col not in df_inference_window.columns:
                logger.warning('  Columna ausente en Feature Store, rellenando con 0: %s', col)
                df_inference_window[col] = 0.0

        X_hoy = df_inference_window[feature_cols].fillna(0).iloc[[-1]]

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
    # PASO 7: Actualizar Log de Predicciones en S3 (append, dedup por date+family)
    # -------------------------------------------------------------------------
    df_results = pd.DataFrame(results_today)

    try:
        response = s3_client.get_object(Bucket=BUCKET_NAME, Key=pred_log_key)
        df_existing_log = pd.read_csv(io.BytesIO(response['Body'].read()))
        df_updated_log  = pd.concat([df_existing_log, df_results], ignore_index=True)
        df_updated_log  = df_updated_log.drop_duplicates(
            subset=['date', 'family'], keep='last'
        ).reset_index(drop=True)
        logger.info('Log de predicciones actualizado (%d registros totales).', len(df_updated_log))
    except s3_client.exceptions.NoSuchKey:
        logger.info('Creando nuevo log de predicciones.')
        df_updated_log = df_results

    csv_buffer = io.StringIO()
    df_updated_log.to_csv(csv_buffer, index=False)
    s3_client.put_object(Bucket=BUCKET_NAME, Key=pred_log_key, Body=csv_buffer.getvalue())

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
