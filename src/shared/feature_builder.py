"""
M�dulo compartido de feature engineering.
Usado por:
  - t2_01_feature_engineering.py  (generar features de T2)
  - t2_04_monthly_retrain.py      (reentrenar con T1+T2)
"""
import pandas as pd
import numpy as np

WINDOWS = {'1h': 6, '6h': 36, '24h': 144, '7d': 1008}
BASELINE_DAYS = 180

FAMILY_SENSORS = {
    'yaw_cable': [
        'nacelle_position', 'nacelle_position_standard_deviation',
        'wind_direction', 'wind_direction_standard_deviation',
        'vane_position_12', 'cable_windings_from_calibration_point',
        'wind_speed_ms', 'power_kw',
        'yaw_error', 'yaw_error_wind', 'cable_rate', 'nacelle_std_ratio',
    ],
    'brake_hydro': [
        'gear_oil_inlet_pressure_bar', 'gear_oil_pump_pressure_bar',
        'gear_oil_inlet_temperature_c', 'gear_oil_temperature_c',
        'generator_rpm_rpm', 'generator_rpm_standard_deviation_rpm',
        'rotor_speed_rpm', 'power_kw',
        'front_bearing_temperature_c', 'rear_bearing_temperature_c',
        'metal_particle_count',
        't_gear_oil_delta', 'pressure_vs_temp', 'metal_particle_rate',
    ],
    'generator': [
        'generator_bearing_front_temperature_c', 'generator_bearing_rear_temperature_c',
        'generator_bearing_front_temperature_max_c', 'generator_bearing_rear_temperature_max_c',
        'nacelle_temperature_c', 'nacelle_ambient_temperature_c',
        'ambient_temperature_converter_c', 'power_kw', 'reactive_power_kvar',
        'power_factor_cosphi', 'stator_temperature_1_c', 'wind_speed_ms',
        't_bearing_delta', 't_rear_bearing_delta', 't_stator_delta',
        't_bearing_diff', 't_stator_bearing_diff',
        'apparent_power_kva', 'reactive_power_ratio',
        't_bearing_delta_roc', 't_stator_roc',
    ],
    'pitch_bat': [
        'motor_current_axis_1_a', 'motor_current_axis_2_a', 'motor_current_axis_3_a',
        'blade_angle_pitch_position_a', 'blade_angle_pitch_position_b', 'blade_angle_pitch_position_c',
        't_motor1_vs_ambient', 't_motor2_vs_ambient', 't_motor3_vs_ambient',
        'power_kw', 'wind_speed_ms',
        'pitch_asymmetry', 'blade_angle_mean', 'motor_current_imbalance',
    ],
}

def add_domain_features(df: pd.DataFrame) -> pd.DataFrame:
    """Añade features calculadas de dominio físico."""
    df = df.copy()
    df['yaw_error']         = (df['nacelle_position'] - df['wind_direction']).abs() % 360
    df['yaw_error']         = df['yaw_error'].apply(lambda x: x if x <= 180 else 360 - x)
    df['yaw_error_wind']    = df['yaw_error'] * df['wind_speed_ms']
    df['cable_rate']        = df['cable_windings_from_calibration_point'].diff(1).fillna(0)
    df['nacelle_std_ratio'] = df['nacelle_position_standard_deviation'] / (df['wind_speed_ms'] + 1e-6)
    df['t_bearing_delta']      = df['generator_bearing_front_temperature_c'] - df['nacelle_ambient_temperature_c']
    df['t_rear_bearing_delta'] = df['generator_bearing_rear_temperature_c']  - df['nacelle_ambient_temperature_c']
    df['t_stator_delta']       = df['stator_temperature_1_c']                - df['nacelle_ambient_temperature_c']
    df['t_gear_oil_delta']     = df['gear_oil_temperature_c']                - df['nacelle_ambient_temperature_c']
    df['t_bearing_diff']       = df['generator_bearing_front_temperature_c'] - df['generator_bearing_rear_temperature_c']
    df['t_stator_bearing_diff']= df['stator_temperature_1_c']                - df['generator_bearing_front_temperature_c']
    df['t_bearing_delta_roc']  = df['t_bearing_delta'].diff(6)
    df['t_stator_roc']         = df['stator_temperature_1_c'].diff(6)
    df['apparent_power_kva']   = (df['power_kw']**2 + df['reactive_power_kvar']**2) ** 0.5
    df['reactive_power_ratio'] = df['reactive_power_kvar'] / (df['apparent_power_kva'] + 1e-6)
    df['pressure_vs_temp']     = df['gear_oil_inlet_pressure_bar'] / (df['gear_oil_inlet_temperature_c'] + 273.15)
    df['metal_particle_rate']  = df['metal_particle_count'].diff(1).fillna(0).clip(lower=0)
    df['t_motor1_vs_ambient']  = df['temperature_motor_axis_1_c'] - df['nacelle_ambient_temperature_c']
    df['t_motor2_vs_ambient']  = df['temperature_motor_axis_2_c'] - df['nacelle_ambient_temperature_c']
    df['t_motor3_vs_ambient']  = df['temperature_motor_axis_3_c'] - df['nacelle_ambient_temperature_c']
    df['pitch_asymmetry']      = (df[['blade_angle_pitch_position_a','blade_angle_pitch_position_b','blade_angle_pitch_position_c']].max(axis=1) -
                                   df[['blade_angle_pitch_position_a','blade_angle_pitch_position_b','blade_angle_pitch_position_c']].min(axis=1))
    df['blade_angle_mean']         = df[['blade_angle_pitch_position_a','blade_angle_pitch_position_b','blade_angle_pitch_position_c']].mean(axis=1)
    df['motor_current_imbalance']  = df[['motor_current_axis_1_a','motor_current_axis_2_a','motor_current_axis_3_a']].std(axis=1)
    return df

def compute_baseline(df: pd.DataFrame) -> tuple[dict, dict]:
    """Calcula baseline (mean y p90) sobre los primeros 180 días."""
    cutoff = df['timestamp'].min() + pd.Timedelta(days=BASELINE_DAYS)
    df_bl  = df[df['timestamp'] < cutoff]
    sensor_cols = [c for c in df.columns
                   if c not in ['timestamp'] and not c.startswith('is_pre_')
                   and not c.startswith('hours_to_') and not c.startswith('hours_since_')
                   and df[c].dtype in [float, 'float64', 'float32']]
    return df_bl[sensor_cols].mean().to_dict(), df_bl[sensor_cols].quantile(0.90).to_dict()

def make_rolling_features(df: pd.DataFrame, sensors: list,
                           baseline_mean: dict, baseline_p90: dict) -> pd.DataFrame:
    """Genera features rolling: mean, std, p95, exceedance, baseline_ratio."""
    feats = {}
    for col in sensors:
        if col not in df.columns:
            continue
        s      = df[col].ffill().fillna(0)
        thresh = baseline_p90.get(col, s.quantile(0.90))
        for wname, w in WINDOWS.items():
            mp   = max(1, w // 3)
            roll = s.rolling(w, min_periods=mp)
            feats[f'{col}__mean_{wname}']   = roll.mean()
            feats[f'{col}__std_{wname}']    = roll.std().fillna(0)
            feats[f'{col}__p95_{wname}']    = roll.quantile(0.95)
            feats[f'{col}__exceed_{wname}'] = s.rolling(w, min_periods=mp).apply(
                lambda x: (x > thresh).mean(), raw=True)
        bm = baseline_mean.get(col, 1.0)
        if abs(bm) > 1e-6:
            feats[f'{col}__baseline_ratio'] = s.rolling(
                WINDOWS['7d'], min_periods=max(1, WINDOWS['7d']//3)
            ).mean() / abs(bm)
    return pd.DataFrame(feats, index=df.index)

def add_temporal_context(df: pd.DataFrame, family: str, fault_times: list) -> pd.DataFrame:
    """Añade hours_since_last_fault (y versión log)."""
    fault_arr   = np.array(fault_times, dtype='datetime64[ns]')
    ts_arr      = df['timestamp'].values.astype('datetime64[ns]')
    hours_since = np.full(len(ts_arr), np.nan)
    for i, ts in enumerate(ts_arr):
        past = fault_arr[fault_arr <= ts]
        hours_since[i] = float((ts - past[-1]) / np.timedelta64(1,'h')) if len(past) > 0 else 8760.0
    df[f'hours_since_last_{family}']     = hours_since
    df[f'hours_since_last_{family}_log'] = np.log1p(hours_since)
    return df