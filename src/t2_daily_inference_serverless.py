import os
import json
import pickle
import logging
import io
import sys
import boto3
import pandas as pd
import numpy as np

# Buscamos las funciones de feature engineering que viajan en tu repositorio
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

# Configuración del Logger optimizado para ver las salidas en AWS CloudWatch
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# Configuración del entorno real de S3
BUCKET_NAME = os.environ.get('AWS_S3_BUCKET', 'ai-driven-cross-generator-transfer-learning')
TURBINE_ID  = 2

FAMILIES = {
    'yaw_cable':   {'alert_h': 48,  'lead_hours': 83},
    'generator':   {'alert_h': 72,  'lead_hours': 127},
    'brake_hydro': {'alert_h': 72,  'lead_hours': 130},
    'pitch_bat':   {'alert_h': 168, 'lead_hours': 295},
}

# 7 días de historial a pasos de 10 minutos = 1008 registros necesarios
STEPS_7D = 1008

def handler(event, context):
    """Punto de entrada nativo (Handler) ejecutado por AWS Lambda."""
    logger.info('=' * 60)
    logger.info('INFERENCIA DIARIA SERVERLESS — Turbina %d', TURBINE_ID)
    logger.info('=' * 60)

    s3_client = boto3.client('s3')
    hoy = pd.Timestamp.now().normalize()
    logger.info('Fecha de ejecución de la inferencia: %s', hoy.date())

    # -------------------------------------------------------------------------
    # PASO 1: Descargar y procesar Telemetría limpia (Capa Bronze Dataset)
    # -------------------------------------------------------------------------
    bronze_telemetry_key = f'bronze/turbine_{TURBINE_ID}_telemetry_clean.parquet/'
    logger.info('Descargando dataset particionado desde S3: s3://%s/%s', BUCKET_NAME, bronze_telemetry_key)
    
    try:
        s3_uri = f"s3://{BUCKET_NAME}/{bronze_telemetry_key}"
        df_all = pd.read_parquet(s3_uri)
        df_all = df_all.sort_values('timestamp').reset_index(drop=True)
        logger.info('Dataset Parquet cargado exitosamente (%d filas totales).', len(df_all))
    except Exception as e:
        logger.error('Error crítico al leer el directorio telemetría en S3: %s', str(e))
        return {"statusCode": 500, "body": f"Error cargando telemetría primaria: {str(e)}"}

    # Filtrar datos simulando entorno de producción diario
    df_hasta_hoy = df_all[df_all['timestamp'] <= hoy].copy()

    if len(df_hasta_hoy) == 0:
        logger.error('Data Lake sin registros válidos hasta la fecha actual (%s).', hoy.date())
        return {"statusCode": 400, "body": "Sin registros en S3 para la fecha simulada."}

    df_window = df_hasta_hoy.tail(STEPS_7D).reset_index(drop=True)
    ultimo_ts = df_window['timestamp'].max()
    primer_ts = df_window['timestamp'].min()
    logger.info('Ventana temporal cargada en memoria: %s -> %s (%d filas)', primer_ts.date(), ultimo_ts.date(), len(df_window))

    # Inyectar características matemáticas de dominio
    df_window = add_domain_features(df_window)

    # -------------------------------------------------------------------------
    # PASO 2: Cargar archivo de Baseline de T2 (JSON) desde S3
    # -------------------------------------------------------------------------
    baseline_key = f'models/turbine_{TURBINE_ID}_baseline.json'
    try:
        response = s3_client.get_object(Bucket=BUCKET_NAME, Key=baseline_key)
        bl = json.loads(response['Body'].read().decode('utf-8'))
        baseline_mean = bl['mean']
        baseline_p90  = bl['p90']
        logger.info('Baseline cargado correctamente desde S3.')
    except Exception as e:
        try:
            logger.warning('Buscando baseline en la raíz del bucket...')
            response = s3_client.get_object(Bucket=BUCKET_NAME, Key=f'turbine_{TURBINE_ID}_baseline.json')
            bl = json.loads(response['Body'].read().decode('utf-8'))
            baseline_mean = bl['mean']
            baseline_p90  = bl['p90']
            logger.info('Baseline cargado correctamente desde la raíz.')
        except Exception:
            # ESTRATEGIA DE RESCATE: Inicializamos un diccionario vacío o valores por defecto
            # para que 'make_rolling_features' no rompa si no encuentra el archivo técnico.
            logger.warning('⚠️ No se encontró el archivo en S3. Usando diccionario de Baseline vacío para la prueba.')
            baseline_mean = {}
            baseline_p90  = {}

    # -------------------------------------------------------------------------
    # PASO 3: Cargar histórico de Logs de Fallos de T2 (CSV) desde S3
    # -------------------------------------------------------------------------
    fault_log_key = f'models/turbine_{TURBINE_ID}_fault_log.csv'
    try:
        response = s3_client.get_object(Bucket=BUCKET_NAME, Key=fault_log_key)
        fault_log = pd.read_csv(io.BytesIO(response['Body'].read()), parse_dates=['timestamp'])
        logger.info('Log de fallos de T2 cargado con éxito desde S3.')
    except s3_client.exceptions.NoSuchKey:
        logger.warning('Log de fallos no encontrado en S3. Inicializando estructura vacía.')
        fault_log = pd.DataFrame(columns=['timestamp', 'family'])

    # DEFINICIÓN CRÍTICA: Ruta de salida mapeada a tu carpeta real de modelos
    pred_log_key = 'models/t2_predictions_log.csv'
    results_today = []

    # -------------------------------------------------------------------------
    # PASO 4: Inferencia iterativa por familia de componentes mecánicos
    # -------------------------------------------------------------------------
    pred_log_key = 'models/t2_predictions_log.csv'
    results_today = []
    
    # Aquí acumularemos la última fila de features calculadas para cada familia
    features_today_dict = {'timestamp': str(ultimo_ts), 'turbine_id': TURBINE_ID}

    for family, cfg in FAMILIES.items():
        logger.info('Procesando subsistema: %s', family)

        model_key = f'models/t1_model_{family}.pkl'
        try:
            response = s3_client.get_object(Bucket=BUCKET_NAME, Key=model_key)
            pipeline = pickle.loads(response['Body'].read())
        except s3_client.exceptions.NoSuchKey:
            logger.warning('Modelo .pkl omitido (no se encuentra en S3): %s', model_key)
            continue

        lgbm         = pipeline['lgbm']
        calibrator   = pipeline['calibrator']
        feature_cols = pipeline['feature_cols']

        # Extraer características rolling
        feats = make_rolling_features(df_window, FAMILY_SENSORS[family], baseline_mean, baseline_p90)

        # Contexto de fallos acumulados
        fault_times = fault_log[fault_log['family'] == family]['timestamp'].tolist()
        if not fault_times:
            t1_targets_key = 'models/t1_fault_targets_grouped.parquet'
            try:
                response = s3_client.get_object(Bucket=BUCKET_NAME, Key=t1_targets_key)
                t1_targets = pd.read_parquet(io.BytesIO(response['Body'].read()))
                fault_times = t1_targets[t1_targets['family'] == family]['timestamp'].tolist()
                logger.info('  [Cold Start] Sincronizando contexto base de T1 (%d fallos)', len(fault_times))
            except Exception as e:
                logger.warning('  No se pudo leer el contexto de T1 (%s). Avanzando sin histórico.', str(e))

        df_feats = pd.concat([df_window[['timestamp']], feats], axis=1)
        df_feats = add_temporal_context(df_feats, family, fault_times)

        for col in feature_cols:
            if col not in df_feats.columns:
                df_feats[col] = 0.0

        # FILTRADO ATÓMICO: Estado actual
        X_hoy = df_feats[feature_cols].fillna(0).iloc[[-1]]

        raw_pred = float(np.clip(lgbm.predict(X_hoy)[0], 0, cfg['lead_hours']))
        cal_pred = float(np.clip(calibrator.predict([raw_pred])[0], 0, cfg['lead_hours']))

        is_alert = cal_pred <= cfg['alert_h']
        logger.info('  Resultado -> Predicción Calibrada: %.1fh | Alerta activa: %s', cal_pred, '🚨 SÍ' if is_alert else '✅ NO')

        # Guardamos en nuestro diccionario todas las columnas calculadas para esta familia específica
        # (Filtramos solo la última fila .iloc[0] para llevárnoslo como clave-valor al diccionario maestro)
        fila_features = df_feats[feature_cols].fillna(0).iloc[[-1]]
        for col in fila_features.columns:
            features_today_dict[col] = float(fila_features[col].values[0])

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
        return {"statusCode": 200, "body": "No se ejecutaron predicciones."}

    # -------------------------------------------------------------------------
    # PASO 5: Actualización del Log de Predicciones Históricas en S3 (Append)
    # -------------------------------------------------------------------------
    df_results = pd.DataFrame(results_today)
    
    try:
        response = s3_client.get_object(Bucket=BUCKET_NAME, Key=pred_log_key)
        df_existing_log = pd.read_csv(io.BytesIO(response['Body'].read()))
        df_updated_log = pd.concat([df_existing_log, df_results], ignore_index=True)
        logger.info('Concatenando resultados en el registro histórico de predicciones S3.')
    except s3_client.exceptions.NoSuchKey:
        logger.info('Iniciando un nuevo log de predicciones maestro en S3.')
        df_updated_log = df_results

    csv_buffer = io.StringIO()
    df_updated_log.to_csv(csv_buffer, index=False)
    s3_client.put_object(Bucket=BUCKET_NAME, Key=pred_log_key, Body=csv_buffer.getvalue())

    # -------------------------------------------------------------------------
    # PASO 6: Guardar la Telemetría Procesada (Feature Store) para Reentrenamiento
    # -------------------------------------------------------------------------
    features_log_key = f'models/turbine_{TURBINE_ID}_features_history.csv'
    
    # Convertimos el diccionario acumulador en un Dataframe limpio de una sola fila
    df_telemetria_procesada_hoy = pd.DataFrame([features_today_dict])
    
    try:
        response = s3_client.get_object(Bucket=BUCKET_NAME, Key=features_log_key)
        df_existing_features = pd.read_csv(io.BytesIO(response['Body'].read()))
        df_updated_features = pd.concat([df_existing_features, df_telemetria_procesada_hoy], ignore_index=True)
        logger.info('Guardando el set completo de variables en el Feature Store de S3.')
    except s3_client.exceptions.NoSuchKey:
        logger.info('Iniciando un nuevo Feature Store histórico (turbine_features_history.csv) en S3.')
        df_updated_features = df_telemetria_procesada_hoy

    feat_csv_buffer = io.StringIO()
    df_updated_features.to_csv(feat_csv_buffer, index=False)
    s3_client.put_object(Bucket=BUCKET_NAME, Key=features_log_key, Body=feat_csv_buffer.getvalue())
    
    logger.info('Pipeline completado con éxito. Registros y Features guardados en S3.')

    return {
        "statusCode": 200,
        "body": json.dumps(results_today)
    }

if __name__ == '__main__':
    print("🚀 Iniciando prueba local del Handler de AWS Lambda...")
    handler({}, None)