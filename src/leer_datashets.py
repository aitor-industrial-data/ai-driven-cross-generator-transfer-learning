import pandas as pd
import scipy.io as sio

# Dataset 1 — leer un CSV
df1 = pd.read_csv('data/GEN_2KVA_4_SALIENT_POLES_FIXED_SPEED/FAULT_GER_ZN_009_TYPE_ABCG_POSEXL000_ACT1000_REA1000_INC000.csv')
print(df1.head())
print(df1.columns.tolist())

# Dataset 2 — leer un .mat
mat = sio.loadmat('data/SCIG-3phase-Dataset/HEALTHY_S1200_T52.mat')
print(mat.keys())