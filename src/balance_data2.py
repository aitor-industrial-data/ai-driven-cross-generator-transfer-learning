import scipy.io as sio
import numpy as np
import os

carpeta = 'data/SCIG-3phase-Dataset'
archivos = [f for f in os.listdir(carpeta) if f.endswith('.mat')]

ok_count = 0
nok_count = 0

for archivo in archivos:
    mat = sio.loadmat(os.path.join(carpeta, archivo))
    fault_relay = mat['Fault_relay'].flatten()
    ok_count += np.sum(fault_relay == 0)
    nok_count += np.sum(fault_relay == 1)

print(f"Dataset 2 — OK: {ok_count}, NOK: {nok_count}")
print(f"Total filas: {ok_count + nok_count}")
print(f"Balance: {ok_count/(ok_count+nok_count)*100:.1f}% OK / {nok_count/(ok_count+nok_count)*100:.1f}% NOK")