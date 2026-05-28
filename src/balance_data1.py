import pandas as pd
import scipy.io as sio
import numpy as np
import os


# ── DATASET 1 ──────────────────────────────────────────────
carpeta = 'data/GEN_2KVA_4_SALIENT_POLES_FIXED_SPEED'
archivos = [f for f in os.listdir(carpeta) if f.endswith('.csv')]

df1_lista = []
for archivo in archivos:
    df = pd.read_csv(os.path.join(carpeta, archivo))
    df1_lista.append(df)

df1 = pd.concat(df1_lista, ignore_index=True)

# Seleccionar solo las columnas útiles con sus nombres reales exactos
columnas_utiles = [
    '1-Time', 
    '2-VGERA', 
    '3-VGERB', 
    '4-VGERC',
    '6-IGERAN', 
    '7-IGERBN', 
    '8-IGERCN',
    '16-Speed (rad/s)',   # <--- Con sus unidades
    '17-Active Power', 
    '19-FAULT '           # <--- Con el espacio al final
]

# Filtrar el DataFrame
df1 = df1[columnas_utiles]

# Renombrar la columna de falla a 'label' de forma segura usando el nombre exacto
df1.rename(columns={'19-FAULT ': 'label'}, inplace=True)

# Comprobar que todo está correcto
print("Dataset 1:", df1.shape)
print(df1['label'].value_counts())