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
        
        # Aquí ya se incluye el timestamp dentro de 'feats'
        feats = add_temporal_context(pd.concat([df[['timestamp']], feats], axis=1),
                                     family, fault_times)

        out_path = os.path.join(SILVER_DIR, f'turbine_{TURBINE_ID}_features_{family}.parquet')
        
        # SOLUCIÓN: Guardamos 'feats' directamente porque ya contiene la columna 'timestamp'
        feats.to_parquet(out_path, index=False)
        print(f'  ✅ {out_path}  [{time.time()-t0:.0f}s]')

if __name__ == '__main__':
    main()