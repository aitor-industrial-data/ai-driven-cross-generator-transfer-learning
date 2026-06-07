"""
Procesa el CSV mensual de fallos de T2 y lo acumula en
fault_targets_grouped_t2.parquet con la misma estructura que T1.

Uso:
  python t2_03_process_fault_log.py --csv /ruta/al/fault_log_mes.csv

El CSV de entrada tiene la estructura original del SCADA:
  Timestamp start, Timestamp end, Duration, Status, Code, Message,
  Comment, Service contract category, IEC category, is_failure_target
"""
import os, argparse
import pandas as pd

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SILVER_DIR = os.path.join(BASE_DIR, 'data', 'silver')

# Mismos códigos que en 03_merge_and_cleaning.ipynb
FAULT_FAMILIES = {
    'yaw_cable':   {'codes': [6052, 6200, 6054, 6120, 6300]},
    'brake_hydro': {'codes': [2125, 5720, 5510, 2000, 1860]},
    'generator':   {'codes': [3000, 2550, 2650, 2655, 2674, 3125, 8400]},
    'pitch_bat':   {'codes': [716, 717, 718, 681, 682, 683, 785, 850]},
}

ALL_TARGET_CODES = {
    code for cfg in FAULT_FAMILIES.values() for code in cfg['codes']
}

def get_family(code):
    for family, cfg in FAULT_FAMILIES.items():
        if code in cfg['codes']:
            return family
    return 'other'

def process_fault_csv(csv_path: str) -> pd.DataFrame:
    """
    Procesa un CSV de fallos con estructura original del SCADA.
    Devuelve un DataFrame agrupado por (timestamp, family) listo
    para acumular en fault_targets_grouped_t2.parquet.
    """
    df = pd.read_csv(csv_path, parse_dates=['Timestamp start'])

    # Filtrar solo eventos marcados como failure target
    targets = df[df['is_failure_target'] == True].copy()
    print(f'  Eventos is_failure_target=True: {len(targets)}')

    if len(targets) == 0:
        print('  Sin eventos de fallo en este CSV.')
        return pd.DataFrame()

    # Filtrar solo los códigos de las 4 familias
    targets = targets[targets['Code'].isin(ALL_TARGET_CODES)].copy()
    print(f'  Eventos en familias entrenables: {len(targets)}')

    # Redondear timestamp hacia abajo al intervalo de 10 minutos
    # (igual que en 03_merge_and_cleaning — floor, no round)
    targets['timestamp'] = targets['Timestamp start'].dt.floor('10min')

    # Asignar familia
    targets['family'] = targets['Code'].apply(get_family)
    targets = targets[targets['family'] != 'other'].copy()

    # Agrupar por (timestamp, familia) — misma lógica que el notebook 03
    grouped = targets.groupby(['timestamp', 'family']).agg(
        Code    = ('Code',    lambda x: ','.join(map(str, sorted(x.unique())))),
        Message = ('Message', lambda x: ' | '.join(x.unique())),
        Status  = ('Status',  'first'),
    ).reset_index()

    counts  = targets.groupby(['timestamp', 'family']).size().reset_index(name='n_events')
    grouped = grouped.merge(counts, on=['timestamp', 'family'])

    print(f'  Eventos agrupados: {len(grouped)}')
    print(f'  Por familia:')
    print(grouped['family'].value_counts().to_string())

    return grouped

def accumulate_faults(new_faults: pd.DataFrame):
    """
    Añade los nuevos fallos al parquet acumulado de T2.
    Elimina duplicados por (timestamp, family) en caso de reprocesar el mismo mes.
    """
    out_path = os.path.join(SILVER_DIR, 'fault_targets_grouped_t2.parquet')

    if os.path.exists(out_path):
        existing = pd.read_parquet(out_path)
        combined = pd.concat([existing, new_faults], ignore_index=True)
        # Eliminar duplicados — por si se reprocesa el mismo CSV
        combined = combined.drop_duplicates(
            subset=['timestamp', 'family'], keep='last'
        ).sort_values('timestamp').reset_index(drop=True)
        print(f'  Fallos previos: {len(existing)} → Total tras añadir: {len(combined)}')
    else:
        combined = new_faults.sort_values('timestamp').reset_index(drop=True)
        print(f'  Primer CSV procesado — creando parquet desde cero: {len(combined)} fallos')

    combined.to_parquet(out_path, index=False)
    print(f'  ✅ Guardado: fault_targets_grouped_t2.parquet')
    print(f'  Rango: {combined["timestamp"].min()} → {combined["timestamp"].max()}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', required=True,
                        help='Ruta al CSV mensual de fallos de T2')
    args = parser.parse_args()

    print(f'Procesando: {args.csv}')
    new_faults = process_fault_csv(args.csv)

    if len(new_faults) > 0:
        accumulate_faults(new_faults)
    else:
        print('Sin fallos nuevos para acumular.')