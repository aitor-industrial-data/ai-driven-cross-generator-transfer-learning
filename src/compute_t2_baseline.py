"""
compute_t2_baseline.py
----------------------
Script puntual para calcular el baseline (mean + p90) de Turbina 2
a partir de su telemetría Bronze en S3 y subir el resultado como
models/turbine_2_baseline.json.

Ejecutar una vez cuando haya suficiente telemetría limpia de T2
(mínimo 2-3 meses). Volver a ejecutar si se quiere actualizar el baseline
con datos más recientes.

Uso:
    python compute_t2_baseline.py

Variables de entorno opcionales:
    AWS_S3_BUCKET   nombre del bucket (por defecto el del proyecto)
    BASELINE_MONTHS número de meses a usar para el cálculo (por defecto 3)
"""

import os
import io
import json
import logging
import sys
import boto3
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src', 'shared'))
try:
    from feature_builder import add_domain_features
except ImportError:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'shared'))
    from feature_builder import add_domain_features

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

BUCKET_NAME     = os.environ.get('AWS_S3_BUCKET', 'ai-driven-cross-generator-transfer-learning')
TURBINE_ID      = 2
BASELINE_MONTHS = int(os.environ.get('BASELINE_MONTHS', 3))

# Columnas que nunca forman parte del baseline (no son sensores físicos)
EXCLUDE_PREFIXES = ('is_pre_', 'hours_to_', 'hours_since_', 'timestamp')


def compute_baseline(df: pd.DataFrame, months: int) -> tuple[dict, dict]:
    """
    Calcula media y p90 sobre los primeros `months` meses de datos.
    Excluye automáticamente columnas de target y contexto temporal.
    """
    cutoff = df['timestamp'].min() + pd.DateOffset(months=months)
    df_bl  = df[df['timestamp'] < cutoff].copy()

    if len(df_bl) == 0:
        raise ValueError(
            f'No hay datos en los primeros {months} meses. '
            f'Rango disponible: {df["timestamp"].min()} → {df["timestamp"].max()}'
        )

    sensor_cols = [
        c for c in df_bl.columns
        if not any(c.startswith(p) for p in EXCLUDE_PREFIXES)
        and df_bl[c].dtype in ['float64', 'float32', 'int64', 'int32']
    ]

    logger.info('Calculando baseline sobre %d filas y %d columnas de sensor.',
                len(df_bl), len(sensor_cols))

    mean_dict = df_bl[sensor_cols].mean().round(6).to_dict()
    p90_dict  = df_bl[sensor_cols].quantile(0.90).round(6).to_dict()

    return mean_dict, p90_dict


def main():
    s3_client = boto3.client('s3')

    # ------------------------------------------------------------------
    # 1. Cargar telemetría Bronze de T2
    # ------------------------------------------------------------------
    bronze_key = f'bronze/turbine_{TURBINE_ID}_telemetry_clean.parquet/'
    logger.info('Cargando Bronze de T2 desde s3://%s/%s', BUCKET_NAME, bronze_key)

    try:
        df = pd.read_parquet(f's3://{BUCKET_NAME}/{bronze_key}')
        df = df.sort_values('timestamp').reset_index(drop=True)
        logger.info('Bronze cargado: %d filas, rango %s → %s',
                    len(df), df['timestamp'].min().date(), df['timestamp'].max().date())
    except Exception as e:
        logger.error('Error cargando Bronze: %s', str(e))
        raise

    # ------------------------------------------------------------------
    # 2. Añadir features de dominio físico (mismas que usa el pipeline)
    # ------------------------------------------------------------------
    logger.info('Calculando features de dominio físico...')
    df = add_domain_features(df)

    # ------------------------------------------------------------------
    # 3. Calcular baseline
    # ------------------------------------------------------------------
    logger.info('Calculando baseline sobre los primeros %d meses...', BASELINE_MONTHS)
    baseline_mean, baseline_p90 = compute_baseline(df, BASELINE_MONTHS)

    logger.info('Baseline calculado: %d sensores.', len(baseline_mean))
    logger.info('Ejemplos mean: %s',
                {k: v for k, v in list(baseline_mean.items())[:5]})
    logger.info('Ejemplos p90:  %s',
                {k: v for k, v in list(baseline_p90.items())[:5]})

    # ------------------------------------------------------------------
    # 4. Serializar y subir a S3
    # ------------------------------------------------------------------
    baseline_payload = {
        'turbine_id':      TURBINE_ID,
        'baseline_months': BASELINE_MONTHS,
        'computed_on':     pd.Timestamp.now().isoformat(),
        'n_rows':          int(len(df[df['timestamp'] < df['timestamp'].min() + pd.DateOffset(months=BASELINE_MONTHS)])),
        'mean':            baseline_mean,
        'p90':             baseline_p90,
    }

    output_key = f'models/turbine_{TURBINE_ID}_baseline.json'
    s3_client.put_object(
        Bucket=BUCKET_NAME,
        Key=output_key,
        Body=json.dumps(baseline_payload, indent=2).encode('utf-8'),
        ContentType='application/json',
    )

    logger.info('✅ Baseline subido correctamente a s3://%s/%s', BUCKET_NAME, output_key)
    logger.info('   Sensores incluidos: %d', len(baseline_mean))
    logger.info('   Filas de referencia: %d', baseline_payload['n_rows'])


if __name__ == '__main__':
    main()
