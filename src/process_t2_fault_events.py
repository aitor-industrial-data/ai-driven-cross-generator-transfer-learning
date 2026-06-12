"""
process_t2_fault_events.py
--------------------------
Procesa el CSV de eventos SCADA de T2, filtra los fallos técnicos reales,
los mapea a familias y hace append al fault log acumulativo en S3.

El fault log resultante (turbine_2_fault_log.csv) es consumido por:
  - t2_daily_inference_serverless.py  → hours_since_last_fault en tiempo real
  - t2_monthly_retrain.py             → etiquetado para reentrenamiento

Uso:
    # Procesar un CSV concreto
    python process_t2_fault_events.py --input /ruta/turbine_2_status_2026_2030.csv

    # Procesar directamente desde S3
    python process_t2_fault_events.py --s3key bronze/turbine_2_status_2026_2030.csv

    # Solo auditar sin escribir nada (dry-run)
    python process_t2_fault_events.py --s3key bronze/turbine_2_status_2026_2030.csv --dry-run
"""

import os
import io
import sys
import logging
import argparse
import boto3
import pandas as pd
from botocore.exceptions import ClientError

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

BUCKET_NAME = os.environ.get('AWS_S3_BUCKET', 'ai-driven-cross-generator-transfer-learning')
TURBINE_ID  = 2

FAULT_LOG_KEY = f'models/turbine_{TURBINE_ID}_fault_log.csv'

# =============================================================================
# CATÁLOGO DE EXCLUSIONES
# Mismos criterios que T1: mantenimiento programado, factores externos,
# acciones operativas humanas.
# =============================================================================
EXCLUDED_CODES = {
    # Mantenimiento programado / pruebas
    710,   # Battery test
    707,   # Stop battery test
    5760,  # Hydraulic oil flushing operation
    5700,  # Max. operation time hydraulic

    # Factores externos / ambientales
    3500,  # Grid loss
    3585,  # Maximum grid frequency
    3590,  # Overvoltage
    6540,  # Icing (anemometer)
    6682,  # Icing (dev. electr. power)
    6690,  # Icing (stop)
    64,    # Max. wind speed
    68,    # Deviation winddirection > 60°
    61,    # Wind < power
    815,   # Rotor overspeed nacelle (parada de seguridad por viento)
    3570,  # Grid error
    3870,  # Overload transformer fan outlet air

    # Acciones operativas humanas
    20,    # Manual stop - on site
    25,    # Manual stop without login
    21,    # Manual stop - remote
    8000,  # Park master stop
    210,   # Manual brake
    7325,  # Time sync. failed
    7324,  # Check time synchronization

    # Sensores externos (anemómetro / veleta) — no predicibles con SCADA
    6525,  # 4-20mA anemometer 2
    6515,  # 4-20mA anemometer 1
    6530,  # Anemometer defect
    6635,  # 4-20 mA vane 2
    6620,  # Vane defect
    6622,  # Vane 2 defect

    # Estructurales / genéricos no asignables
    4510,  # Tower oscillation Y level 1
    4520,  # Tower oscillation X level 1
    4530,  # Tower oscillation Y level 2
    4540,  # Tower oscillation X level 2
    4500,  # Tower resonance
    59,    # Max. acceleration
    100,   # Safety chain open (demasiado genérico)
    5000,  # Breakdown obstacle light
    440,   # Repeating error BP 0
    2950,  # Lightning protection defect
    7057,  # Heating/fan top box faulty
}

# =============================================================================
# MAPEO DE CÓDIGOS A FAMILIAS
# Basado en el catálogo de T1 + auditoría de códigos nuevos en T2.
# =============================================================================
FAMILY_CODES = {
    'yaw_cable': [
        6052,  # High yaw motor current
        6200,  # Cable autounwind
        6054,  # Easy yaw
        6120,  # (T1, puede aparecer en futuro)
        6300,  # (T1, puede aparecer en futuro)
        6005,  # Overload yaw motor 1&3
        6130,  # Yaw speed high
    ],
    'brake_hydro': [
        2125,  # Timeout brake closed
        5720,  # Brake accumulator defect
        5510,  # (T1, puede aparecer en futuro)
        2000,  # Brake pads worn
        1860,  # (T1, puede aparecer en futuro)
        1070,  # Drive train monitor level 2
        1065,  # Drive train monitor level 1
        1050,  # Drivetrain oscillations
        1800,  # Overload gear oil pump
        1620,  # Implausible gear speed
    ],
    'generator': [
        3000,  # Frequency converter not ready
        2550,  # Overload generator fan 1
        2650,  # Overload generator fan 2
        2655,  # Overload generator fan 3
        2674,  # Overload generator heating
        3125,  # Timeout ready for connection
        8400,  # Comm. failure FPM
        3110,  # Frequency converter error
        3210,  # Frequency converter load rejection
        2810,  # Service generator brushes
    ],
    'pitch_bat': [
        716,   # Battery charge cycle axis 1 error
        717,   # Battery charge cycle axis 2 error
        718,   # Battery charge cycle axis 3 error
        681,   # Limit switch error 95° axis 1
        682,   # Limit switch error 95° axis 2
        683,   # Limit switch error 95° axis 3
        785,   # Error brake resistor CHP
        850,   # Error lubrication pump pitch
        692,   # Pitch run-away (hub box v.>=4)
        720,   # Pitch batteries charging cycle
        725,   # Battery voltage axis 1
        675,   # Pitch measuring system 1><2
    ],
}

# Índice inverso: código → familia (para lookup rápido)
CODE_TO_FAMILY = {
    code: family
    for family, codes in FAMILY_CODES.items()
    for code in codes
}


def load_events_csv(path: str) -> pd.DataFrame:
    """Carga el CSV de eventos SCADA desde ruta local."""
    df = pd.read_csv(
        path,
        parse_dates=['Timestamp start', 'Timestamp end'],
        dayfirst=False,
    )
    df = df.rename(columns={'Timestamp start': 'timestamp_start',
                             'Timestamp end':   'timestamp_end'})
    logger.info('CSV cargado: %d filas totales.', len(df))
    return df


def load_events_csv_from_s3(s3_client, key: str) -> pd.DataFrame:
    """Carga el CSV de eventos SCADA desde S3."""
    response = s3_client.get_object(Bucket=BUCKET_NAME, Key=key)
    df = pd.read_csv(
        io.BytesIO(response['Body'].read()),
        parse_dates=['Timestamp start', 'Timestamp end'],
        dayfirst=False,
    )
    df = df.rename(columns={'Timestamp start': 'timestamp_start',
                             'Timestamp end':   'timestamp_end'})
    logger.info('CSV cargado desde S3 (%s): %d filas totales.', key, len(df))
    return df


def process_events(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filtra y mapea eventos a familias.

    Pasos:
    1. Descartar Informational y Communication
    2. Descartar códigos de exclusión (externos, operativos, mantenimiento)
    3. Mapear código → familia
    4. Descartar códigos sin familia asignada (ruido no clasificado)
    5. Deduplicar: si hay varios eventos del mismo código en ±10 min, quedarse
       con el primero (agrupación como en T1)
    """
    n0 = len(df)

    # 1. Solo Stop y Warning
    df = df[df['Status'].isin(['Stop', 'Warning'])].copy()
    logger.info('Tras filtro Stop/Warning: %d filas (-%d)', len(df), n0 - len(df))

    # 2. Excluir códigos de ruido
    n1 = len(df)
    df = df[~df['Code'].isin(EXCLUDED_CODES)].copy()
    logger.info('Tras exclusión de códigos de ruido: %d filas (-%d)', len(df), n1 - len(df))

    # 3. Mapear a familia
    df['family'] = df['Code'].map(CODE_TO_FAMILY)

    # 4. Descartar sin familia
    n2 = len(df)
    df = df[df['family'].notna()].copy()
    unmapped = n2 - len(df)
    if unmapped > 0:
        logger.warning('%d códigos sin familia asignada descartados.', unmapped)

    # 5. Usar timestamp_start como timestamp canónico del fallo
    df['timestamp'] = pd.to_datetime(df['timestamp_start'])

    # 6. Deduplicar: mismo código en ventana de 10 min → quedarse con el primero
    df = df.sort_values('timestamp').reset_index(drop=True)
    df['_bucket'] = df['timestamp'].dt.floor('10min')
    df = df.drop_duplicates(subset=['Code', '_bucket'], keep='first')
    df = df.drop(columns=['_bucket'])

    # Columnas de salida: misma estructura que t1_fault_log.csv
    result = df[['timestamp', 'family', 'Code', 'Message', 'Status']].copy()
    result = result.rename(columns={'Code': 'code', 'Message': 'message',
                                    'Status': 'status'})
    result = result.sort_values('timestamp').reset_index(drop=True)

    logger.info('Fallos técnicos válidos extraídos: %d', len(result))
    logger.info('Distribución por familia:')
    for family, count in result['family'].value_counts().items():
        logger.info('  %-15s %d', family, count)

    return result


def load_existing_fault_log(s3_client) -> pd.DataFrame | None:
    """Carga el fault log acumulativo existente en S3."""
    try:
        response = s3_client.get_object(Bucket=BUCKET_NAME, Key=FAULT_LOG_KEY)
        df = pd.read_csv(io.BytesIO(response['Body'].read()), parse_dates=['timestamp'])
        logger.info('Fault log existente cargado: %d registros.', len(df))
        return df
    except ClientError as e:
        if e.response['Error']['Code'] == 'NoSuchKey':
            logger.info('Fault log no existe aún. Se creará.')
            return None
        raise


def append_to_fault_log(s3_client, df_new: pd.DataFrame, dry_run: bool = False) -> pd.DataFrame:
    """
    Hace append de los nuevos fallos al fault log acumulativo.
    Deduplicación por timestamp + family + code para idempotencia.
    """
    df_existing = load_existing_fault_log(s3_client)

    if df_existing is not None:
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        df_combined = df_combined.drop_duplicates(
            subset=['timestamp', 'family', 'code'], keep='first'
        ).sort_values('timestamp').reset_index(drop=True)
        n_new = len(df_combined) - len(df_existing)
        logger.info('Registros nuevos añadidos al fault log: %d', n_new)
        logger.info('Fault log total tras merge: %d registros.', len(df_combined))
    else:
        df_combined = df_new.copy()
        logger.info('Fault log creado con %d registros.', len(df_combined))

    if dry_run:
        logger.info('[DRY-RUN] No se escribe nada en S3.')
        return df_combined

    csv_buffer = io.StringIO()
    df_combined.to_csv(csv_buffer, index=False)
    s3_client.put_object(
        Bucket=BUCKET_NAME,
        Key=FAULT_LOG_KEY,
        Body=csv_buffer.getvalue(),
    )
    logger.info('Fault log guardado en s3://%s/%s', BUCKET_NAME, FAULT_LOG_KEY)
    return df_combined


def main():
    parser = argparse.ArgumentParser(description='Procesa eventos SCADA de T2 y actualiza fault log.')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--input',  help='Ruta local al CSV de eventos')
    group.add_argument('--s3key',  help='Clave S3 del CSV de eventos (sin s3://bucket/)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Auditar sin escribir en S3')
    args = parser.parse_args()

    s3_client = boto3.client('s3')

    # Cargar CSV de eventos
    if args.input:
        df_events = load_events_csv(args.input)
    else:
        df_events = load_events_csv_from_s3(s3_client, args.s3key)

    # Procesar
    df_faults = process_events(df_events)

    if len(df_faults) == 0:
        logger.warning('No se encontraron fallos técnicos válidos en el CSV. Nada que guardar.')
        return

    # Append al fault log acumulativo
    df_log = append_to_fault_log(s3_client, df_faults, dry_run=args.dry_run)

    # Resumen final
    logger.info('=' * 55)
    logger.info('RESUMEN FINAL DEL FAULT LOG ACUMULATIVO')
    logger.info('=' * 55)
    logger.info('Total registros: %d', len(df_log))
    logger.info('Rango:           %s → %s',
                df_log['timestamp'].min().date(),
                df_log['timestamp'].max().date())
    for family, count in df_log['family'].value_counts().items():
        logger.info('  %-15s %d fallos', family, count)


if __name__ == '__main__':
    main()