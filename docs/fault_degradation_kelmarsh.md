## CATÁLOGO DE FALLOS: TIEMPOS Y SENSORES DEFINITIVOS
 
> Basado en los datos reales del fault_log.csv y el análisis de nulos de la telemetría.
> Los sensores marcados con ⛔ tienen >40% de nulos — NO usar como feature primaria.
> Los sensores marcados con ⚠️ tienen 10–40% de nulos — usar con imputación.
 
---
 
### FAMILIA 1 — YAW / CABLE ✅ ENTRENAR PRIMERO
**572 total · 217 eventos · Distribuidos 2018–2021**
 
| Código | Mensaje | Eventos | Status |
|--------|---------|---------|--------|
| 6052 | High yaw motor current | 132 | Warning |
| 6200 | Cable autounwind | 66 | Stop |
| 6054 | Easy yaw | 17 | Warning |
| 6120 | Uncontrolled yaw movement | 1 | Stop |
| 6300 | Yaw error | 1 | Stop |
 
**Lead time recomendado para etiquetado: 168 horas (7 días)**
**Mi estimación de cuándo empieza la señal real: 5–10 días antes**
 
El 6200 (Cable autounwind) es casi determinista — `Cable_windings_from_calibration_point`
sube linealmente hasta el umbral. No hace falta ML para ese, pero incluirlo en la familia
refuerza el modelo. El 6052 tiene señal clara en la std de nacelle_position y el error
de alineación nacelle-viento.
 
**Sensores a usar (todos con <5% nulos):**
 
| Columna exacta CSV | Nulos | Rol |
|---|---|---|
| `Nacelle_position` | 1.36% | Posición nacelle — señal primaria |
| `Nacelle_position_Standard_deviation` | 2.60% | Inestabilidad orientación |
| `Wind_direction` | 1.36% | Dirección viento |
| `Wind_direction_Standard_deviation` | 2.60% | Turbulencia direccional |
| `Vane_position_12` | 3.71% | Error alineación directo |
| `Cable_windings_from_calibration_point` | 3.71% | Señal DIRECTA 6200 — contador vueltas |
| `Wind_speed_ms` | 1.36% | Variable de control |
| `Power_kW` | 1.36% | Variable de control |
 
**Sensores BAD — NO usar:**
 
| Columna | Nulos | Por qué se descarta |
|---|---|---|
| `Cable_windings_from_calibration_point_Max` | 53.41% | Mitad del tiempo vacío |
| `Cable_windings_from_calibration_point_Min` | 53.41% | Mitad del tiempo vacío |
| `Vane_position_12_StdDev` | 53.41% | Mitad del tiempo vacío |
 
**Feature engineering:**
- `yaw_error = abs(Nacelle_position - Wind_direction)` → media y slope en ventanas 1h/6h/24h
- `cable_rate = delta(Cable_windings_from_calibration_point) / 10min` → pendiente acumulada
- `nacelle_std_ratio = Nacelle_position_std / Wind_speed_ms` → esfuerzo normalizado
---
 
### FAMILIA 2 — FRENO / HIDRÁULICO ✅ ENTRENAR
**63 eventos · Distribuidos uniformemente 2018–2021**
 
| Código | Mensaje | Eventos | Status |
|--------|---------|---------|--------|
| 2125 | Timeout brake closed | 31 | Warning |
| 5720 | Brake accumulator defect | 22 | Warning |
| 5510 | Low hydraulic pressure | 5 | Stop |
| 2000 | Brake pads worn | 2 | Warning |
| 1860 | Oil filter gear choked | 3 | Warning |
 
**Lead time recomendado para etiquetado: 120 horas (5 días)**
**Mi estimación de cuándo empieza la señal real: 3–7 días antes**
 
La presión hidráulica (`Gear_oil_inlet_pressure_bar`) baja gradualmente días antes.
El patrón de frenado (RPM no baja a 0 en tiempo esperado) es la señal más discriminante
para 2125 — hay que extraer la dinámica de eventos de parada.
 
IMPORTANTE: Las columnas `_Max` y `_Min` de gear_oil_inlet_pressure tienen 52% de nulos
— usar solo el valor medio que tiene 1.36% de nulos.
 
**Sensores a usar:**
 
| Columna exacta CSV | Nulos | Rol |
|---|---|---|
| `Gear_oil_inlet_pressure_bar` | 1.36% | Señal primaria — presión |
| `Gear_oil_pump_pressure_bar` | 1.36% | Presión de la bomba |
| `Gear_oil_inlet_temperature_C` | 3.72% | Temperatura aceite — normalizar presión |
| `Gear_oil_temperature_C` | 3.71% | Temperatura del aceite |
| `Generator_RPM_RPM` | 1.36% | Dinámica de frenado |
| `Generator_RPM_Standard_deviation_RPM` | 2.60% | Inestabilidad durante frenado |
| `Rotor_speed_RPM` | 1.36% | Velocidad de rotor |
| `Power_kW` | 1.36% | Detectar eventos de parada |
| `Front_bearing_temperature_C` | 3.71% | Estado mecánico general |
| `Rear_bearing_temperature_C` | 3.71% | Estado mecánico general |
| `Metal_particle_count` | 0.00% | Desgaste mecánico — sin nulos |
 
**Sensores BAD — NO usar:**
 
| Columna | Nulos |
|---|---|
| `Gear_oil_inlet_pressure_Max_bar` | 52.29% |
| `Gear_oil_inlet_pressure_Min_bar` | 52.29% |
| `Gear_oil_pump_pressure_Max_bar` | 52.29% |
| `Gear_oil_pump_pressure_Min_bar` | 52.29% |
 
**Feature engineering:**
- `pressure_vs_temp = Gear_oil_inlet_pressure_bar / (Gear_oil_inlet_temperature_C + 273)` → caída de presión normalizada por viscosidad
- `rpm_decel_slope` → pendiente negativa de RPM en ventanas donde Power baja de 100→0 kW
- `metal_particle_rate = delta(Metal_particle_count) / tiempo` → tasa de desgaste
---
 
### FAMILIA 3 — GENERADOR / CONVERTIDOR / FANS ✅ ENTRENAR
**98 eventos · Distribuidos uniformemente 2018–2021 (decreciente)**
 
| Código | Mensaje | Eventos | Status |
|--------|---------|---------|--------|
| 8400 | Comm. failure FPM | 24 | Warning |
| 3000 | Frequency converter not ready | 21 | Stop |
| 2550 | Overload generator fan 1 | 13 | Warning |
| 2650 | Overload generator fan 2 | 11 | Warning |
| 2655 | Overload generator fan 3 | 11 | Warning |
| 2674 | Overload generator heating | 11 | Warning |
| 3125 | Timeout ready for connection | 7 | Stop |
 
**Lead time recomendado para etiquetado: 120 horas (5 días)**
**Mi estimación de cuándo empieza la señal real: 3–7 días antes**
 
Fans 1+2+3 tienen exactamente los mismos sensores y misma física — se entrenan como
un solo target `fan_overload`. La señal es la temperatura de rodamientos del generador
subiendo sobre la curva esperada para esa potencia.
 
CRÍTICO: `Ambient_temperature_converter_Max_C` y `_StdDev` tienen 52% de nulos.
Usar SOLO `Ambient_temperature_converter_C` que tiene 1.36% de nulos.
 
**Sensores a usar:**
 
| Columna exacta CSV | Nulos | Rol |
|---|---|---|
| `Generator_bearing_front_temperature_C` | 1.36% | Señal primaria — T rodamiento delantero |
| `Generator_bearing_rear_temperature_C` | 1.36% | Señal primaria — T rodamiento trasero |
| `Generator_bearing_front_temperature_Max_C` | 7.15% | Pico térmico 10 min |
| `Generator_bearing_rear_temperature_Max_C` | 7.15% | Pico térmico 10 min |
| `Nacelle_temperature_C` | 3.71% | T interior nacelle |
| `Nacelle_temperature_Max_C` | 8.26% | Pico nacelle |
| `Nacelle_ambient_temperature_C` | 3.71% | T exterior referencia |
| `Ambient_temperature_converter_C` | 1.36% | T entrada convertidor |
| `Power_kW` | 1.36% | Normalizar temperatura por carga |
| `Reactive_power_kvar` | 1.36% | Carga del convertidor |
| `Power_factor_cosphi` | 1.36% | Degradación del convertidor |
| `Stator_temperature_1_C` | 1.36% | Temperatura estátor |
| `Wind_speed_ms` | 1.36% | Variable de control |
 
**Sensores BAD — NO usar:**
 
| Columna | Nulos |
|---|---|
| `Ambient_temperature_converter_Max_C` | 52.29% |
| `Ambient_temperature_converter_StdDev_C` | 52.29% |
 
**Feature engineering:**
- `T_bearing_vs_expected = Generator_bearing_front_temperature_C - f(Power_kW)` → residuo sobre curva térmica esperada
- `delta_T_nacelle = Nacelle_temperature_C - Nacelle_ambient_temperature_C` → gradiente interior/exterior
- `cosphi_slope` → pendiente del factor de potencia en ventana 24h
---
 
### FAMILIA 4 — PITCH / BATERÍAS ✅ ENTRENAR (con cautela)
**71 eventos · Distribuidos 2018–2021 (salvo 2019 que tiene solo 4)**
 
| Código | Mensaje | Eventos | Status |
|--------|---------|---------|--------|
| 716 | Battery charge cycle axis 1 error | 17 | Warning |
| 717 | Battery charge cycle axis 2 error | 17 | Warning |
| 718 | Battery charge cycle axis 3 error | 11 | Warning |
| 785 | Error brake resistor CHP | 6 | Warning |
| 850 | Error lubrication pump pitch | 4 | Warning |
| 681/682/683 | Limit switch error 95° axis 1/2/3 | 7 | Warning |
| 675 | Pitch measuring system 1><2 | 3 | Warning |
 
**Lead time recomendado para etiquetado: 336 horas (14 días)**
**Mi estimación de cuándo empieza la señal real: 2–4 semanas antes**
 
Los 716/717/718 son estacionales — ocurren en invierno (baterías frías). El esfuerzo
del motor pitch aumenta semanas antes. Con ventana larga de 14 días, los 45 eventos de
baterías dan señal suficiente aunque sean pocos.
 
ATENCIÓN: Las columnas `_Standard_deviation` de blade angles tienen 32% de nulos.
Usarlas con imputación por forward fill durante períodos de operación normal.
 
**Sensores a usar:**
 
| Columna exacta CSV | Nulos | Rol |
|---|---|---|
| `Motor_current_axis_1_A` | 1.36% | Esfuerzo motor pitch blade A |
| `Motor_current_axis_2_A` | 1.36% | Esfuerzo motor pitch blade B |
| `Motor_current_axis_3_A` | 1.36% | Esfuerzo motor pitch blade C |
| `Blade_angle_pitch_position_A` | 3.71% | Ángulo blade A |
| `Blade_angle_pitch_position_B` | 3.71% | Ángulo blade B |
| `Blade_angle_pitch_position_C` | 3.71% | Ángulo blade C |
| `Temperature_motor_axis_1_C` | 1.36% | T motor pitch A |
| `Temperature_motor_axis_2_C` | 1.36% | T motor pitch B |
| `Nacelle_ambient_temperature_C` | 3.71% | T exterior — correlaciona con batería fría |
| `Power_kW` | 1.36% | Variable de control |
| `Wind_speed_ms` | 1.36% | Variable de control |
 
**Sensores BAD — NO usar como primarios:**
 
| Columna | Nulos | Alternativa |
|---|---|---|
| `Motor_current_axis_1_Max_A` | 52.30% | Usar rolling max sobre valor medio |
| `Motor_current_axis_1_StdDev_A` | 52.30% | Calcular con rolling std |
| `Blade_angle_pitch_position_A_Standard_deviation` | 32.65% | ⚠️ Imputar o calcular rolling |
 
**Feature engineering:**
- `motor_effort_ratio = Motor_current_axis_1_A / abs(delta(Blade_angle_A))` → corriente por grado de movimiento
- `pitch_asymmetry = max(A,B,C) - min(A,B,C)` → divergencia entre ejes
- `T_ambient_slope` → pendiente de temperatura exterior en 24h (predice ciclo batería fría)
---
 
### DRIVETRAIN ❌ NO ENTRENAR
**1 único evento en todo el período**
 
Código 1070 — 1 sola ocurrencia en 2019. Sin datos suficientes para ningún modelo.
Documentar como "regla de alerta manual" basada en Metal_particle_count y
Drive_train_acceleration_mmss pero no entrenar ML.
 
---
 
## PARTE 2 — RESUMEN EJECUTIVO DE VIABILIDAD
 
| Familia | Eventos | Años con datos | Sensores OK | Veredicto |
|---------|---------|---------------|-------------|-----------|
| Yaw/Cable | 217 | 2018/19/20/21 | ✅ Todos disponibles | ✅ ENTRENAR |
| Freno/Hidráulico | 63 | 2018/19/20/21 | ✅ Todos disponibles | ✅ ENTRENAR |
| Generador/Fans | 98 | 2018/19/20/21 | ✅ Todos disponibles | ✅ ENTRENAR |
| Pitch/Baterías | 71 | 2018/20/21 | ⚠️ Max/StdDev con 32-52% nulos | ✅ ENTRENAR (cuidado) |
| Drivetrain | 1 | Solo 2019 | N/A | ❌ SKIP |
 
**Orden de trabajo recomendado: Yaw → Generador → Freno → Pitch**


---
 
## FALLOS DESCARTADOS — JUSTIFICACIÓN TÉCNICA
 
### ❌ Familia: Sensores anemómetro / vane (6525, 6635, 6530)
 
**Eventos:** 86 en total  
**Distribución temporal:** 100% concentrados en diciembre 2021 (un único mes de cuatro años)
 
**Por qué no se entrena:**
 
Los 86 eventos no representan un patrón recurrente de degradación sino un único incidente de hardware ocurrido en un período de 31 días. Un modelo de ML entrenado con estos datos memorizaría las condiciones ambientales y operativas de ese mes concreto (velocidad de viento, temperatura, carga) sin aprender nada generalizable. En producción fallaría en cualquier condición distinta a diciembre 2021.
 
Adicionalmente, la naturaleza del fallo (señal 4–20 mA fuera de rango) no produce degradación progresiva detectable semanas antes — el fallo es eléctrico y puede ser abrupto (conector suelto, humedad en la caja de conexiones). No hay "rampa de degradación" que el modelo pueda aprender.
 
**Alternativa recomendada:** regla determinista. Si `abs(Wind_speed_Sensor_1 - Wind_speed_Sensor_2) > 1.5 m/s` durante más de 3 intervalos consecutivos de 10 min → alerta de sensor divergente. Sin ML, sin datos de entrenamiento, igualmente efectivo.
 
**Columnas eliminadas del dataset por ser exclusivas de esta familia:**
- `Wind_speed_Sensor_1_ms` — se mantiene como variable de control general
- `Wind_speed_Sensor_2_ms` — se mantiene como variable de control general
- No hay columnas únicas que eliminar; los sensores de viento son usados por otras familias como variables de control
---
 
### ❌ Familia: Torre / Vibración (4510, 4540, 59)
 
**Eventos:** 26 en total (11 × 4510, 8 × 4540, 7 × 59)  
**Distribución temporal:** presentes en 2018–2021, concentrados en 2019–2020
 
**Por qué no se entrena:**
 
**Problema 1 — Insuficiencia estadística:** 26 eventos para un modelo con ventana de 3 días y ~144 pasos de 10 min por ventana es insuficiente. LightGBM necesita al menos 30–50 positivos en el conjunto de entrenamiento para aprender representaciones robustas. Con un split 80/20 temporal, el train tendría ~18–20 positivos — por debajo del umbral mínimo.
 
**Problema 2 — Resolución temporal incompatible:** Las oscilaciones de torre (códigos 4510/4540) son eventos dinámicos que ocurren en segundos o minutos. La resolución del dataset SCADA es de 10 minutos — cada fila es la media de 600 segundos de datos. Las columnas `Tower_Acceleration_X_mmss` y `Tower_Acceleration_y_mmss` son medias de 10 minutos: un pico de aceleración de 0.5 segundos queda diluido en 600 puntos y puede ser imperceptible. Para detectar oscilaciones de torre se necesitan datos de alta frecuencia (≥1 Hz), no SCADA de 10 min. Adicionalmente las columnas Max/Min/StdDev de Tower Acceleration — que sí capturarían los picos — tienen 53.41% de valores nulos, inutilizables como features primarias.
 
**Problema 3 — Señal insuficiente en lo disponible:** El código 59 (Max. acceleration) es un warning de aceleración máxima puntual, no una degradación progresiva. No hay ventana de degradación de días — el sistema de control detecta el pico y emite el warning casi simultáneamente.
 
**Alternativa recomendada:** monitorización de reglas de umbral sobre `Tower_Acceleration_X_mmss` y `Tower_Acceleration_y_mmss` con alertas en tiempo real. Si la media de 10 min supera 2σ del baseline histórico para esa velocidad de viento → alerta operativa. Más adecuado que ML para este tipo de fallo.
 
**Columnas A eliminar del dataset de entrenamiento** (no aportan a ninguna familia activa):
- `Tower_Acceleration_X_mmss`
- `Tower_Acceleration_y_mmss`
- `Tower_Acceleration_X_Max_mmss` (53.41% nulos)
- `Tower_Acceleration_X_Min_mmss` (53.41% nulos)
- `Tower_Acceleration_Y_Max_mmss` (53.41% nulos)
- `Tower_Acceleration_Y_Min_mmss` (53.41% nulos)
- `Tower_Acceleration_X_StdDev_mmss` (53.41% nulos)
- `Tower_Acceleration_Y_StdDev_mmss` (53.41% nulos)
 
---
 
### ❌ Familia: Drivetrain (1070)
 
**Eventos:** 1 único evento en 2019
 
**Por qué no se entrena:** Con un único evento histórico es matemáticamente imposible entrenar un modelo supervisado. No hay mínimo estadístico que lo justifique.
 
**Alternativa recomendada:** vigilancia manual de `Metal_particle_count` (0% nulos) y `Drive_train_acceleration_mmss`. Si el contador de partículas metálicas supera el percentil 95 del histórico → escalar a revisión técnica presencial.
 