"""
generador_base.py
=================
Genera datos históricos de 1 año (1 muestra/minuto) para las 3 máquinas
de una línea de embotellado. Los datos son nominales con ruido gaussiano
realista: sin averías, pero con variaciones propias de operación real.

Realismo incorporado:
  - Turnos de trabajo (6:00-22:00 L-V, 6:00-14:00 S, cerrado D)
  - Arranque y parada diaria (transitorios realistas)
  - Variación de carga a lo largo del turno
  - Temperatura ambiente con ciclo estacional y diurno
  - Degradación basal muy lenta (envejecimiento normal de componentes)
  - Paradas programadas de mantenimiento preventivo (1 vez/mes)
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os

# ── Semilla para reproducibilidad ────────────────────────────────────────────
SEED = 42
rng = np.random.default_rng(SEED)

# ── Período de simulación ─────────────────────────────────────────────────────
START = datetime(2024, 1, 1, 0, 0, 0)
END   = datetime(2024, 12, 31, 23, 59, 0)
FREQ  = "1min"

OUTPUT_DIR = "/mnt/user-data/outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════════
# UTILIDADES COMUNES
# ═══════════════════════════════════════════════════════════════════════════════

def build_timestamps():
    return pd.date_range(start=START, end=END, freq=FREQ)


def machine_state_series(timestamps):
    """
    Devuelve array con el estado de la máquina minuto a minuto.
    RUNNING / IDLE (ralentí entre lotes) / STOPPED (fuera de turno o mant.)
    Turno: L-V 06:00-22:00 · S 06:00-14:00 · D cerrado.
    Mantenimiento preventivo: primer lunes de cada mes, 08:00-12:00.
    """
    n = len(timestamps)
    state = np.full(n, "STOPPED", dtype=object)

    # Mantenimientos preventivos: primer lunes de cada mes
    pm_slots = set()
    for month in range(1, 13):
        # Primer lunes del mes
        d = datetime(2024, month, 1)
        while d.weekday() != 0:          # 0 = lunes
            d += timedelta(days=1)
        pm_start = d.replace(hour=8, minute=0)
        pm_end   = d.replace(hour=12, minute=0)
        t = pm_start
        while t <= pm_end:
            pm_slots.add(t)
            t += timedelta(minutes=1)

    for i, ts in enumerate(timestamps):
        if ts in pm_slots:
            state[i] = "STOPPED"        # mantenimiento preventivo
            continue
        dow  = ts.weekday()             # 0=L … 6=D
        hour = ts.hour
        minute = ts.minute
        if dow == 6:                    # domingo cerrado
            continue
        if dow == 5:                    # sábado medio turno
            if 6 <= hour < 14:
                state[i] = "RUNNING"
        else:                           # lunes-viernes
            if 6 <= hour < 22:
                state[i] = "RUNNING"

    # Ralentí: 5 min después del arranque y 5 min antes de parada
    for i in range(1, n):
        if state[i] == "RUNNING" and state[i-1] == "STOPPED":
            for j in range(i, min(i+5, n)):
                state[j] = "IDLE"
        if state[i] == "STOPPED" and state[i-1] == "RUNNING":
            for j in range(max(0, i-5), i):
                state[j] = "IDLE"

    return state


def ambient_temp(timestamps):
    """
    Temperatura ambiente con ciclo anual (invierno frío, verano caliente)
    y ciclo diurno. Base Pamplona: 5°C enero, 22°C julio.
    """
    n = len(timestamps)
    doy = np.array([ts.timetuple().tm_yday for ts in timestamps])
    hour = np.array([ts.hour + ts.minute / 60 for ts in timestamps])
    seasonal = 13.5 + 8.5 * np.sin(2 * np.pi * (doy - 80) / 365)
    diurnal  = 3.0  * np.sin(2 * np.pi * (hour - 6) / 24)
    noise    = rng.normal(0, 0.4, n)
    return seasonal + diurnal + noise


def load_factor(timestamps, state):
    """
    Factor de carga 0-1 que varía a lo largo del turno.
    Simula variación de demanda: arranque bajo, sube, pico a media mañana,
    baja ligeramente tras el almuerzo, sube de nuevo, baja al final.
    """
    n = len(timestamps)
    lf = np.zeros(n)
    for i, ts in enumerate(timestamps):
        if state[i] == "STOPPED":
            lf[i] = 0.0
        elif state[i] == "IDLE":
            lf[i] = rng.uniform(0.05, 0.15)
        else:
            h = ts.hour + ts.minute / 60
            # Curva de carga típica de turno industrial
            if 6 <= h < 7:
                base = 0.5 + (h - 6) * 0.3        # arranque
            elif 7 <= h < 10:
                base = 0.80 + (h - 7) * 0.05      # subida mañana
            elif 10 <= h < 13:
                base = 0.92                         # pico
            elif 13 <= h < 14:
                base = 0.75                         # almuerzo
            elif 14 <= h < 17:
                base = 0.88                         # tarde
            elif 17 <= h < 20:
                base = 0.82                         # bajada suave
            else:
                base = 0.65 - (h - 20) * 0.05      # final turno
            lf[i] = np.clip(base + rng.normal(0, 0.03), 0.3, 1.0)
    return lf


def aging_factor(timestamps, rate_per_year=0.03):
    """
    Degradación basal muy lenta: simula el envejecimiento normal.
    Al final del año los valores nominales son ~3% peores que al inicio.
    """
    total_minutes = (END - START).total_seconds() / 60
    elapsed = np.array([(ts - START).total_seconds() / 60
                        for ts in timestamps])
    return 1.0 + rate_per_year * (elapsed / total_minutes)


# ═══════════════════════════════════════════════════════════════════════════════
# MÁQUINA 1 — CINTA TRANSPORTADORA
# Motor trifásico asíncrono 5.5 kW · IE3
# ═══════════════════════════════════════════════════════════════════════════════

def generate_machine01(timestamps, state, t_amb, load, aging):
    n = len(timestamps)
    rows = []

    # Parámetros nominales
    I_NOMINAL   = 11.5    # A
    VIB_BASELINE = 0.55   # mm/s (zona A nueva)
    SPEED_SYNC  = 1500.0  # rpm (motor 4 polos 50 Hz)
    SLIP_NOMINAL = 0.03   # deslizamiento nominal
    POWER_NOMINAL = 5500  # W

    for i, ts in enumerate(timestamps):
        s = state[i]
        lf = load[i]
        ag = aging[i]
        ta = t_amb[i]

        if s == "STOPPED":
            rows.append({
                "timestamp": ts,
                "machine_state": s,
                "current_A_L1": 0.0,
                "current_A_L2": 0.0,
                "current_A_L3": 0.0,
                "phase_imbalance_pct": 0.0,
                "vibration_mms": rng.uniform(0.02, 0.06),   # ruido mecánico residual
                "motor_temp_C": ta + rng.normal(2, 0.5),    # enfriándose al ambiente
                "speed_rpm": 0.0,
                "power_kW": 0.0,
                "fault_type": "NONE",
                "fault_active": 0,
            })
            continue

        # Corriente de fase: depende de carga y envejecimiento
        I_base = I_NOMINAL * lf * ag
        # Las 3 fases tienen pequeño desequilibrio natural (<1%)
        imb = rng.normal(0, 0.004)   # desequilibrio natural <0.5%
        I_L1 = I_base * (1 + imb)           + rng.normal(0, 0.08)
        I_L2 = I_base * (1 - imb/2)         + rng.normal(0, 0.08)
        I_L3 = I_base * (1 - imb/2)         + rng.normal(0, 0.08)
        I_L1, I_L2, I_L3 = max(0.1, I_L1), max(0.1, I_L2), max(0.1, I_L3)

        I_avg = (I_L1 + I_L2 + I_L3) / 3
        phase_imb = (max(I_L1, I_L2, I_L3) - min(I_L1, I_L2, I_L3)) / I_avg * 100

        # Velocidad: cae con la carga (deslizamiento)
        slip = SLIP_NOMINAL * lf * ag
        speed = SPEED_SYNC * (1 - slip) + rng.normal(0, 2)

        # Vibración: crece muy lentamente con el envejecimiento
        vib = VIB_BASELINE * ag * lf + rng.normal(0, 0.04)
        vib = max(0.05, vib)

        # Temperatura motor: función de carga y temperatura ambiente
        # Térmica: T_motor = T_amb + ΔT_carga
        delta_T = 35 * lf * ag           # rise térmico por carga
        if s == "IDLE":
            delta_T *= 0.2
        motor_temp = ta + delta_T + rng.normal(0, 0.8)

        # Potencia eléctrica
        cos_phi = 0.86
        V_LINE = 400.0
        power = (3**0.5 * V_LINE * I_avg * cos_phi) / 1000   # kW

        rows.append({
            "timestamp": ts,
            "machine_state": s,
            "current_A_L1": round(I_L1, 3),
            "current_A_L2": round(I_L2, 3),
            "current_A_L3": round(I_L3, 3),
            "phase_imbalance_pct": round(phase_imb, 3),
            "vibration_mms": round(vib, 3),
            "motor_temp_C": round(motor_temp, 2),
            "speed_rpm": round(speed, 1),
            "power_kW": round(power, 3),
            "fault_type": "NONE",
            "fault_active": 0,
        })

    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════════
# MÁQUINA 2 — HORNO DE SELLADO / RETRACTILADO
# Resistencias Calrod 10 kW · Control PID · Setpoint 160°C
# ═══════════════════════════════════════════════════════════════════════════════

def generate_machine02(timestamps, state, t_amb, aging):
    n = len(timestamps)
    rows = []

    SETPOINT     = 160.0    # °C
    POWER_MAX    = 10.0     # kW (4 resistencias × 2.5 kW)
    FAN_NOMINAL  = 1420     # rpm
    # PID simplificado: estado interno
    temp_current = 20.0     # empieza a temperatura ambiente
    pid_integral = 0.0
    KP, KI = 0.8, 0.02

    for i, ts in enumerate(timestamps):
        s = state[i]
        ag = aging[i]
        ta = t_amb[i]

        if s == "STOPPED":
            # El horno se enfría hacia T_ambiente
            temp_current = temp_current * 0.985 + ta * 0.015 + rng.normal(0, 0.3)
            temp_current = max(ta, temp_current)
            pid_integral = 0.0
            fan_rpm = rng.uniform(0, 30)    # ventilador parado
            rows.append({
                "timestamp": ts,
                "machine_state": s,
                "temp_chamber_C": round(temp_current, 2),
                "power_kW": 0.0,
                "pid_output_pct": 0.0,
                "fan_rpm": round(fan_rpm, 0),
                "temp_power_ratio": 0.0,
                "fault_type": "NONE",
                "fault_active": 0,
            })
            continue

        # PID simplificado minuto a minuto
        error = SETPOINT - temp_current
        pid_integral += error * KI
        pid_integral = np.clip(pid_integral, 0, 60)
        pid_out = np.clip(KP * error + pid_integral, 0, 100)   # 0-100%

        # Potencia activa de las resistencias
        # Envejecimiento: degradación muy lenta del aislamiento (+2% consumo/año)
        power = (pid_out / 100) * POWER_MAX * ag + rng.normal(0, 0.05)
        power = max(0, power)

        # Temperatura responde con inercia térmica
        # Calor aportado - pérdidas al ambiente
        heat_in  = power * 8.0                     # coeficiente de calentamiento
        heat_out = (temp_current - ta) * 0.18      # pérdidas proporcionales a ΔT
        delta_T_min = (heat_in - heat_out) / 60    # por minuto
        temp_current += delta_T_min + rng.normal(0, 0.25)
        temp_current = max(ta, min(temp_current, 210))

        # Fan: velocidad nominal con pequeña variación
        fan_rpm = FAN_NOMINAL * ag + rng.normal(0, 15)
        fan_rpm = max(0, fan_rpm)

        # Ratio temperatura/potencia (feature derivada clave)
        ratio = temp_current / power if power > 0.5 else 0.0

        rows.append({
            "timestamp": ts,
            "machine_state": s,
            "temp_chamber_C": round(temp_current, 2),
            "power_kW": round(power, 3),
            "pid_output_pct": round(pid_out, 1),
            "fan_rpm": round(fan_rpm, 0),
            "temp_power_ratio": round(ratio, 2),
            "fault_type": "NONE",
            "fault_active": 0,
        })

    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════════
# MÁQUINA 3 — BRAZO ROBOT DE PALETIZADO
# Sistema neumático 6 bar + servomotores 400 V
# ═══════════════════════════════════════════════════════════════════════════════

def generate_machine03(timestamps, state, t_amb, load, aging):
    n = len(timestamps)
    rows = []

    P_NOMINAL  = 6.0    # bar
    V_NOMINAL  = 400.0  # V
    TORQUE_NOM = 55.0   # % par nominal en ciclo típico
    cycle_accum = 0

    for i, ts in enumerate(timestamps):
        s = state[i]
        lf = load[i]
        ag = aging[i]

        if s == "STOPPED":
            # Presión se mantiene en el depósito (sistema cerrado en pausa)
            pressure = P_NOMINAL * 0.95 + rng.normal(0, 0.04)
            rows.append({
                "timestamp": ts,
                "machine_state": s,
                "pressure_bar": round(pressure, 3),
                "voltage_V_L1": round(V_NOMINAL + rng.normal(0, 1.5), 2),
                "voltage_V_L2": round(V_NOMINAL + rng.normal(0, 1.5), 2),
                "voltage_V_L3": round(V_NOMINAL + rng.normal(0, 1.5), 2),
                "voltage_imbalance_pct": round(abs(rng.normal(0, 0.3)), 3),
                "servo_torque_pct": 0.0,
                "cycle_count": cycle_accum,
                "alarm_code": 0,
                "fault_type": "NONE",
                "fault_active": 0,
            })
            continue

        # Presión neumática: cae durante ciclo de pinza, se recupera
        # Variación dinámica realista minuto a minuto
        cycles_this_min = int(lf * 8 + rng.normal(0, 0.5))   # ~4-8 ciclos/min
        cycles_this_min = max(0, cycles_this_min)
        cycle_accum += cycles_this_min

        # Caída de presión proporcional a ciclos, recuperada por compresor
        p_drop = cycles_this_min * 0.012 * ag
        pressure = P_NOMINAL - p_drop + rng.normal(0, 0.05)
        pressure = np.clip(pressure, 4.5, 7.5)

        # Voltaje de red: variación natural EN 50160 (400V ±10%)
        v_base = V_NOMINAL + rng.normal(0, 3.0)
        # Pequeño desequilibrio natural entre fases (<0.5%)
        imb_v = rng.normal(0, 0.002)
        V_L1 = v_base * (1 + imb_v)  + rng.normal(0, 0.8)
        V_L2 = v_base * (1 - imb_v)  + rng.normal(0, 0.8)
        V_L3 = v_base               + rng.normal(0, 0.8)
        V_avg = (V_L1 + V_L2 + V_L3) / 3
        v_imb = (max(V_L1,V_L2,V_L3) - min(V_L1,V_L2,V_L3)) / V_avg * 100

        # Par de servomotores: sube con carga y envejecimiento del reductor
        torque = TORQUE_NOM * lf * ag + rng.normal(0, 1.5)
        torque = np.clip(torque, 5, 95)

        rows.append({
            "timestamp": ts,
            "machine_state": s,
            "pressure_bar": round(pressure, 3),
            "voltage_V_L1": round(V_L1, 2),
            "voltage_V_L2": round(V_L2, 2),
            "voltage_V_L3": round(V_L3, 2),
            "voltage_imbalance_pct": round(v_imb, 3),
            "servo_torque_pct": round(torque, 2),
            "cycle_count": cycle_accum,
            "alarm_code": 0,
            "fault_type": "NONE",
            "fault_active": 0,
        })

    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("Generando timestamps...")
    timestamps = build_timestamps()
    n = len(timestamps)
    print(f"  Total: {n:,} muestras ({n/60/24:.0f} días)")

    print("Calculando variables comunes...")
    state  = machine_state_series(timestamps)
    t_amb  = ambient_temp(timestamps)
    load   = load_factor(timestamps, state)
    aging  = aging_factor(timestamps)

    running_pct = (state == "RUNNING").sum() / n * 100
    print(f"  RUNNING: {running_pct:.1f}% · IDLE: {(state=='IDLE').sum()/n*100:.1f}% · STOPPED: {(state=='STOPPED').sum()/n*100:.1f}%")

    print("\nGenerando Máquina 1 — Cinta transportadora...")
    df1 = generate_machine01(timestamps, state, t_amb, load, aging)
    out1 = os.path.join(OUTPUT_DIR, "machine_01_conveyor.csv")
    df1.to_csv(out1, index=False)
    print(f"  Guardado: {out1}  ({len(df1):,} filas)")

    print("Generando Máquina 2 — Horno retractilado...")
    df2 = generate_machine02(timestamps, state, t_amb, aging)
    out2 = os.path.join(OUTPUT_DIR, "machine_02_oven.csv")
    df2.to_csv(out2, index=False)
    print(f"  Guardado: {out2}  ({len(df2):,} filas)")

    print("Generando Máquina 3 — Robot paletizador...")
    df3 = generate_machine03(timestamps, state, t_amb, load, aging)
    out3 = os.path.join(OUTPUT_DIR, "machine_03_robot.csv")
    df3.to_csv(out3, index=False)
    print(f"  Guardado: {out3}  ({len(df3):,} filas)")

    # ── Estadísticas de validación ────────────────────────────────────────────
    print("\n── Validación Máquina 1 (filas RUNNING) ──")
    r1 = df1[df1.machine_state == "RUNNING"]
    for col in ["current_A_L1", "vibration_mms", "motor_temp_C", "power_kW"]:
        print(f"  {col:25s}  min={r1[col].min():.2f}  mean={r1[col].mean():.2f}  max={r1[col].max():.2f}")

    print("\n── Validación Máquina 2 (filas RUNNING) ──")
    r2 = df2[df2.machine_state == "RUNNING"]
    for col in ["temp_chamber_C", "power_kW", "pid_output_pct", "fan_rpm"]:
        print(f"  {col:25s}  min={r2[col].min():.2f}  mean={r2[col].mean():.2f}  max={r2[col].max():.2f}")

    print("\n── Validación Máquina 3 (filas RUNNING) ──")
    r3 = df3[df3.machine_state == "RUNNING"]
    for col in ["pressure_bar", "voltage_V_L1", "servo_torque_pct"]:
        print(f"  {col:25s}  min={r3[col].min():.2f}  mean={r3[col].mean():.2f}  max={r3[col].max():.2f}")

    print("\n✓ Generación completada. 3 CSV listos para el inyector de eventos.")

if __name__ == "__main__":
    main()
