# Análisis de Degradación por Código de Fallo — Kelmarsh SCADA (Senvion MM92)

**Dataset**: Kelmarsh Wind Farm SCADA · Resolución 10 min · 303 columnas  
**Turbina**: Senvion MM92 (2.05 MW, rotor 92.5 m, DFIG + convertidor + pitch DC)  
**Uso previsto**: Entrenamiento de modelo predictivo de fallos

---

## Nota metodológica

Cada fallo tiene tres tiempos indicados:

- **Ventana de degradación**: cuánto tiempo antes del evento hay señal detectable en el SCADA
- **Lead time recomendado para etiquetado**: cuánto tiempo antes del fallo se etiquetan los datos como "pre-fallo" para entrenar el modelo (más conservador que la ventana máxima)
- **Sensores exactos del CSV**: nombre exacto de la columna tal como aparece en el dataset

---

## Tabla completa de fallos

---

### Código 59 — Max. acceleration | Warning | Operating states

| Campo | Valor |
|---|---|
| Status | Warning |
| Ventana de degradación | 2–4 semanas |
| **Lead time recomendado** | **14 días antes del evento** |
| Señal más temprana | Incremento progresivo de picos de aceleración bajo misma velocidad de viento. Varianza de RPM aumenta. La std deviation del drive train acceleration es la primera en dispararse. |

**Sensores SCADA necesarios:**

| Columna exacta en CSV | Rol en la predicción |
|---|---|
| `Drive train acceleration (mm/ss)` | Señal primaria — picos y tendencia |
| `Drive train acceleration, Min (mm/ss)` | Asimetría del perfil de vibración |
| `Generator RPM (RPM)` | Correlacionar vibración con velocidad |
| `Generator RPM, Standard deviation (RPM)` | Varianza de RPM como indicador de inestabilidad |
| `Wind speed (m/s)` | Variable de control para normalizar |
| `Power (kW)` | Normalizar vibración respecto a carga |

**Feature engineering sugerido**: ratio `Drive train acceleration / Power` para detectar vibración anómala relativa a la carga operativa.

---

### Código 675 — Pitch measuring system 1><2 | Warning | Warnings

| Campo | Valor |
|---|---|
| Status | Warning |
| Ventana de degradación | 1–3 semanas |
| **Lead time recomendado** | **10 días antes del evento** |
| Señal más temprana | Diferencia creciente entre ángulo blade A vs B/C. La corriente del motor pitch empieza a oscilar antes del error. |

**Sensores SCADA necesarios:**

| Columna exacta en CSV | Rol |
|---|---|
| `Blade angle (pitch position) A (°)` | Señal primaria divergencia |
| `Blade angle (pitch position) B (°)` | Referencia comparación |
| `Blade angle (pitch position) C (°)` | Referencia comparación |
| `Blade angle (pitch position) A, Standard deviation (°)` | Inestabilidad de lectura |
| `Blade angle (pitch position) B, Standard deviation (°)` | Inestabilidad de lectura |
| `Motor current axis 1 (A)` | Esfuerzo motor pitch blade A |
| `Motor current axis 2 (A)` | Esfuerzo motor pitch blade B |
| `Power (kW)` | Variable de control |

**Feature engineering**: `abs(BladeA - BladeB)` y `abs(BladeA - BladeC)` — la divergencia entre ejes es la señal más limpia.

---

### Código 681 — Limit switch error 95° axis 1 | Warning | Warnings

| Campo | Valor |
|---|---|
| Status | Warning |
| Ventana de degradación | Horas – 2 días |
| **Lead time recomendado** | **12 horas antes del evento** |
| Señal más temprana | Blade A se aproxima progresivamente al límite de 95°. Pico de corriente del motor axis 1. |

**Sensores SCADA necesarios:**

| Columna exacta en CSV | Rol |
|---|---|
| `Blade angle (pitch position) A (°)` | Señal primaria — proximidad al límite |
| `Blade angle (pitch position) A, Max (°)` | Captura picos de 10 min |
| `Motor current axis 1 (A)` | Esfuerzo mecánico antes del disparo |
| `Motor current axis 1, Max (A)` | Pico de corriente |

---

### Código 682 — Limit switch error 95° axis 2 | Warning | Warnings

| Campo | Valor |
|---|---|
| Status | Warning |
| Ventana de degradación | Horas – 2 días |
| **Lead time recomendado** | **12 horas antes del evento** |

**Sensores SCADA necesarios:**

| Columna exacta en CSV | Rol |
|---|---|
| `Blade angle (pitch position) B (°)` | Señal primaria |
| `Blade angle (pitch position) B, Max (°)` | Captura picos |
| `Motor current axis 2 (A)` | Esfuerzo mecánico |
| `Motor current axis 2, Max (A)` | Pico corriente |

---

### Código 683 — Limit switch error 95° axis 3 | Warning | Warnings

| Campo | Valor |
|---|---|
| Status | Warning |
| Ventana de degradación | Horas – 2 días |
| **Lead time recomendado** | **12 horas antes del evento** |

**Sensores SCADA necesarios:**

| Columna exacta en CSV | Rol |
|---|---|
| `Blade angle (pitch position) C (°)` | Señal primaria |
| `Blade angle (pitch position) C, Max (°)` | Captura picos |
| `Motor current axis 3 (A)` | Esfuerzo mecánico |
| `Motor current axis 3, Max (A)` | Pico corriente |

---

### Código 697 — Timeout B sensor active | Warning | Warnings

| Campo | Valor |
|---|---|
| Status | Warning |
| Ventana de degradación | 1–5 días |
| **Lead time recomendado** | **3 días antes del evento** |
| Señal más temprana | Gaps esporádicos o lecturas NaN en blade B antes del timeout completo. Patrón de dropout intermitente. |

**Sensores SCADA necesarios:**

| Columna exacta en CSV | Rol |
|---|---|
| `Blade angle (pitch position) B (°)` | Detectar NaN / gaps / spikes |
| `Blade angle (pitch position) B, Standard deviation (°)` | StdDev elevada = inestabilidad de señal |
| `Motor current axis 2 (A)` | Esfuerzo motor axis B |

**Feature engineering**: contar NaNs o outliers (>3σ) en ventana de 24h en blade B.

---

### Código 716 — Battery charge cycle axis 1 error | Warning | Warnings

| Campo | Valor |
|---|---|
| Status | Warning |
| Ventana de degradación | 2–8 semanas |
| **Lead time recomendado** | **21 días antes del evento** |
| Señal más temprana | Respuesta del pitch axis 1 más lenta (corriente alta, movimiento lento de blade). La batería de emergencia no es visible directamente en SCADA, pero la velocidad de respuesta del motor la delata. |

**Sensores SCADA necesarios:**

| Columna exacta en CSV | Rol |
|---|---|
| `Motor current axis 1 (A)` | Señal primaria — corriente alta para mover el mismo ángulo |
| `Motor current axis 1, Max (A)` | Picos de esfuerzo |
| `Motor current axis 1, StdDev (A)` | Irregularidad de corriente |
| `Blade angle (pitch position) A (°)` | Velocidad de cambio de ángulo |
| `Blade angle (pitch position) A, Standard deviation (°)` | Inestabilidad de movimiento |
| `Temperature motor axis 1 (°C)` | Temperatura del motor que refleja esfuerzo |

**Feature engineering**: ratio `Motor current axis 1 / delta(Blade angle A)` — si para mover el mismo ángulo se necesita más corriente, la batería/motor está degradada.

---

### Código 717 — Battery charge cycle axis 2 error | Warning | Warnings

| Campo | Valor |
|---|---|
| Status | Warning |
| Ventana de degradación | 2–8 semanas |
| **Lead time recomendado** | **21 días antes del evento** |

**Sensores SCADA necesarios:**

| Columna exacta en CSV | Rol |
|---|---|
| `Motor current axis 2 (A)` | Señal primaria |
| `Motor current axis 2, Max (A)` | Picos |
| `Motor current axis 2, StdDev (A)` | Irregularidad |
| `Blade angle (pitch position) B (°)` | Velocidad de respuesta |
| `Temperature motor axis 2 (°C)` | Temperatura motor |

---

### Código 718 — Battery charge cycle axis 3 error | Warning | Warnings

| Campo | Valor |
|---|---|
| Status | Warning |
| Ventana de degradación | 2–8 semanas |
| **Lead time recomendado** | **21 días antes del evento** |

**Sensores SCADA necesarios:**

| Columna exacta en CSV | Rol |
|---|---|
| `Motor current axis 3 (A)` | Señal primaria |
| `Motor current axis 3, Max (A)` | Picos |
| `Motor current axis 3, StdDev (A)` | Irregularidad |
| `Blade angle (pitch position) C (°)` | Velocidad de respuesta |
| `Temperature motor axis 3 (°C)` | Temperatura motor |

---

### Código 785 — Error brake resistor CHP | Warning | Warnings

| Campo | Valor |
|---|---|
| Status | Warning |
| Ventana de degradación | 1–7 días |
| **Lead time recomendado** | **4 días antes del evento** |
| Señal más temprana | Temperatura del convertidor sube por encima del patrón normal para esa potencia. Ocurre más frecuentemente tras paradas y rearranques frecuentes. |

**Sensores SCADA necesarios:**

| Columna exacta en CSV | Rol |
|---|---|
| `Ambient temperature (converter) (°C)` | Señal primaria — temperatura de entrada al convertidor |
| `Ambient temperature (converter), Max (°C)` | Pico térmico |
| `Reactive power (kvar)` | Carga del convertidor |
| `Power (kW)` | Variable de control para normalizar temperatura |

---

### Código 1070 — Drive train monitor level 2 | Stop | Warnings

| Campo | Valor |
|---|---|
| Status | **STOP** |
| Ventana de degradación | 2–6 semanas |
| **Lead time recomendado** | **28 días antes del evento** |
| Señal más temprana | El más predecible del dataset. Aumento gradual de vibración + temperatura de aceite fuera de curva + partículas metálicas en aceite. Todas las señales convergiendo semanas antes. |

**Sensores SCADA necesarios:**

| Columna exacta en CSV | Rol |
|---|---|
| `Drive train acceleration (mm/ss)` | **Señal primaria y más temprana** |
| `Gear oil temperature (°C)` | Temperatura aceite caja |
| `Gear oil temperature, Max (°C)` | Pico térmico |
| `Gear oil inlet temperature (°C)` | Temperatura entrada aceite |
| `Gear oil inlet pressure (bar)` | Presión aceite |
| `Gear oil pump pressure (bar)` | Presión bomba |
| `Front bearing temperature (°C)` | Temperatura rodamiento delantero |
| `Rear bearing temperature (°C)` | Temperatura rodamiento trasero |
| `Metal particle count` | **Señal más discriminante** — partículas en aceite |
| `Metal particle count counter` | Contador acumulado |
| `Generator RPM (RPM)` | RPM para normalizar |
| `Power (kW)` | Carga para normalizar temperatura |

**Feature engineering**: crear un score compuesto: `(Gear oil temp - expected_for_power) + (Drive train acceleration - baseline) + metal_particles_delta`. La convergencia de los tres es altamente predictiva.

---

### Código 1810 — Overload gear heating | Warning | Electrical error

| Campo | Valor |
|---|---|
| Status | Warning |
| Ventana de degradación | 1–4 semanas |
| **Lead time recomendado** | **14 días antes del evento** |
| Señal más temprana | Temperatura de aceite de caja sube respecto a la curva esperada para esa potencia. El delta-T entre temperatura real y esperada aumenta progresivamente. |

**Sensores SCADA necesarios:**

| Columna exacta en CSV | Rol |
|---|---|
| `Gear oil temperature (°C)` | Señal primaria |
| `Gear oil temperature, Max (°C)` | Pico |
| `Gear oil inlet temperature (°C)` | Temperatura diferencial entrada/salida |
| `Gear oil inlet pressure (bar)` | Presión del circuito de lubricación |
| `Nacelle temperature (°C)` | Temperatura ambiente nacelle como referencia |
| `Nacelle ambient temperature (°C)` | Temperatura exterior referencia |
| `Power (kW)` | Variable de control — normalizar temperatura |

---

### Código 2000 — Brake pads worn | Warning | Warnings

| Campo | Valor |
|---|---|
| Status | Warning |
| Ventana de degradación | 1–3 meses |
| **Lead time recomendado** | **45 días antes del evento** |
| Señal más temprana | El tiempo de frenado aumenta progresivamente — en el SCADA de 10 min se observa como RPM que no baja a cero en el tiempo esperado tras paradas. Requiere analizar eventos de parada individuales. |

**Sensores SCADA necesarios:**

| Columna exacta en CSV | Rol |
|---|---|
| `Generator RPM (RPM)` | Velocidad de deceleración en paradas |
| `Generator RPM, Min (RPM)` | ¿Llega a cero? |
| `Generator RPM, Standard deviation (RPM)` | Estabilidad durante frenado |
| `Gear oil inlet pressure (bar)` | Presión hidráulica del freno |
| `Gear oil pump pressure (bar)` | Presión bomba |
| `Power (kW)` | Detectar eventos de parada (Power → 0) |

**Feature engineering**: extraer ventanas de evento de parada (Power > 100 kW → Power = 0) y medir la pendiente de deceleración de RPM. Una pendiente que se aplana progresivamente señala desgaste de pastillas.

---

### Código 2125 — Timeout brake closed | Warning | Warnings

| Campo | Valor |
|---|---|
| Status | Warning |
| Ventana de degradación | Minutos – horas |
| **Lead time recomendado** | **2 horas antes del evento** (muy escaso — depende de 2000) |
| Señal más temprana | RPM no baja a cero en el tiempo esperado tras orden de parada. Señal casi simultánea. Predecir desde código 2000. |

**Sensores SCADA necesarios:**

| Columna exacta en CSV | Rol |
|---|---|
| `Generator RPM (RPM)` | No alcanza cero |
| `Generator RPM, Min (RPM)` | Confirmar no llegó a cero |
| `Gear oil inlet pressure (bar)` | Presión durante frenado |

---

### Código 2550 — Overload generator fan 1 | Warning | Warnings

| Campo | Valor |
|---|---|
| Status | Warning |
| Ventana de degradación | 2–7 días |
| **Lead time recomendado** | **5 días antes del evento** |
| Señal más temprana | Temperatura de rodamientos del generador sube sin que la potencia lo justifique. El ventilador trabaja más para compensar → sobrecarga. |

**Sensores SCADA necesarios:**

| Columna exacta en CSV | Rol |
|---|---|
| `Generator bearing front temperature (°C)` | Señal primaria |
| `Generator bearing rear temperature (°C)` | Señal primaria |
| `Generator bearing front temperature, Max (°C)` | Pico |
| `Generator bearing rear temperature, Max (°C)` | Pico |
| `Nacelle temperature (°C)` | Temperatura nacelle como referencia ambiental |
| `Nacelle ambient temperature (°C)` | Temperatura exterior |
| `Power (kW)` | Normalizar temperatura por carga |

---

### Código 2650 — Overload generator fan 2 | Warning | Warnings

| Campo | Valor |
|---|---|
| Status | Warning |
| Ventana de degradación | 2–7 días |
| **Lead time recomendado** | **5 días antes del evento** |

**Sensores SCADA necesarios:** Idénticos a código 2550. Verificar si ambos fallos (2550 y 2650) aparecen correlacionados en el historial de eventos.

---

### Código 2674 — Overload generator heating | Warning | Warnings

| Campo | Valor |
|---|---|
| Status | Warning |
| Ventana de degradación | 3–10 días |
| **Lead time recomendado** | **7 días antes del evento** |
| Señal más temprana | La diferencia entre temperatura nacelle y temperatura ambiente exterior diverge progresivamente. La calefacción trabaja en exceso → precede la sobrecarga. |

**Sensores SCADA necesarios:**

| Columna exacta en CSV | Rol |
|---|---|
| `Nacelle temperature (°C)` | Temperatura interior nacelle |
| `Nacelle temperature, Max (°C)` | Pico |
| `Nacelle ambient temperature (°C)` | Temperatura exterior |
| `Generator bearing front temperature (°C)` | Estado térmico generador |
| `Generator bearing rear temperature (°C)` | Estado térmico generador |

**Feature engineering**: `delta_T = Nacelle temperature - Nacelle ambient temperature`. Una deriva positiva sostenida es la señal.

---

### Código 3110 — Frequency converter error | Stop | Generator and Converter errors

| Campo | Valor |
|---|---|
| Status | **STOP** |
| Ventana de degradación | 2–5 días |
| **Lead time recomendado** | **3 días antes del evento** |
| Señal más temprana | Temperatura de entrada de aire al convertidor sube. Desequilibrio de fases (corrientes L1/L2/L3 divergen). Power factor inestable. Dos o más señales simultáneas son altamente predictivas. |

**Sensores SCADA necesarios:**

| Columna exacta en CSV | Rol |
|---|---|
| `Ambient temperature (converter) (°C)` | Señal primaria — térmica convertidor |
| `Ambient temperature (converter), Max (°C)` | Pico térmico |
| `Current L1 / U (A)` | Corriente fase 1 |
| `Current L2 / V (A)` | Corriente fase 2 |
| `Current L3 / W (A)` | Corriente fase 3 |
| `Power factor (cosphi)` | Factor de potencia |
| `Power factor (cosphi), Standard deviation` | Inestabilidad del cosphi |
| `Reactive power (kvar)` | Potencia reactiva |
| `Grid voltage (V)` | Tensión de red |
| `Grid frequency (Hz)` | Frecuencia de red |

**Feature engineering**: `std(L1, L2, L3)` para detectar desequilibrio de fases; combinarlo con temperatura convertidor.

---

### Código 3160 — Cable overload | Warning | Warnings

| Campo | Valor |
|---|---|
| Status | Warning |
| Ventana de degradación | 1–2 meses |
| **Lead time recomendado** | **30 días antes del evento** |
| Señal más temprana | El contador de vueltas del cable se acumula progresivamente. Cuando se aproxima al límite sin resetear, la sobrecarga es inminente. La señal es determinista y muy predecible. |

**Sensores SCADA necesarios:**

| Columna exacta en CSV | Rol |
|---|---|
| `Cable windings from calibration point` | **Señal directa y única** — contador de vueltas |
| `Cable windings from calibration point, Max` | Máximo en ventana 10 min |
| `Cable windings from calibration point, Min` | Mínimo en ventana 10 min |
| `Nacelle position (°)` | Orientación de la nacelle — dirección de movimiento |
| `Nacelle position, Standard deviation (°)` | Frecuencia de giros de yaw |

**Nota**: Este es el fallo más fácil de predecir del dataset. El cable winding counter es un indicador directo y sin ambigüedad. Un umbral de alerta a 70-80% del límite máximo tiene precisión casi perfecta.

---

### Código 3205 — PT100 converter inlet temperature defect | Warning | Warnings

| Campo | Valor |
|---|---|
| Status | Warning |
| Ventana de degradación | 1–4 días |
| **Lead time recomendado** | **2 días antes del evento** |
| Señal más temprana | La temperatura de entrada al convertidor empieza a divergir del valor esperado comparada con la temperatura ambiente nacelle. Aparecen lecturas erráticas (spikes, NaN) antes del fallo total del sensor. |

**Sensores SCADA necesarios:**

| Columna exacta en CSV | Rol |
|---|---|
| `Ambient temperature (converter) (°C)` | Señal con fallo — buscar outliers |
| `Ambient temperature (converter), Max (°C)` | Detectar spikes |
| `Ambient temperature (converter), Min (°C)` | Detectar dropouts |
| `Ambient temperature (converter), StdDev (°C)` | Inestabilidad de señal — sube antes del fallo |
| `Nacelle ambient temperature (°C)` | Referencia para detectar divergencia |

---

### Código 3220 — Reduced power converter | Warning | Warnings

| Campo | Valor |
|---|---|
| Status | Warning |
| Ventana de degradación | 1–3 semanas |
| **Lead time recomendado** | **14 días antes del evento** |
| Señal más temprana | La potencia activa queda sistemáticamente por debajo de la curva de potencia esperada para la misma velocidad de viento. El cosphi se degrada. Es una de las señales más limpias y tempranas. |

**Sensores SCADA necesarios:**

| Columna exacta en CSV | Rol |
|---|---|
| `Power (kW)` | Señal primaria — desvío de curva de potencia |
| `Wind speed (m/s)` | Variable de control |
| `Power factor (cosphi)` | Degradación del factor de potencia |
| `Power factor (cosphi), Standard deviation` | Inestabilidad |
| `Reactive power (kvar)` | Potencia reactiva elevada = convertidor compensando |
| `Ambient temperature (converter) (°C)` | Estado térmico del convertidor |

**Feature engineering**: Calcular residuo `Power_actual - Power_expected(wind_speed)` usando la curva de potencia de la turbina. Un residuo negativo sostenido es la señal más limpia.

---

### Código 3870 — Overload transformer fan outlet air | Warning | Warnings

| Campo | Valor |
|---|---|
| Status | Warning |
| Ventana de degradación | 2–7 días |
| **Lead time recomendado** | **5 días antes del evento** |
| Señal más temprana | Temperatura de nacelle sube más de lo esperado relativo a la potencia generada y temperatura exterior. El ventilador del transformador trabaja en exceso. |

**Sensores SCADA necesarios:**

| Columna exacta en CSV | Rol |
|---|---|
| `Transformer temperature (°C)` | Temperatura directa del transformador |
| `Transformer temperature, Max (°C)` | Pico térmico |
| `Transformer cell temperature (°C)` | Temperatura celda transformador |
| `Nacelle temperature (°C)` | Temperatura nacelle |
| `Nacelle ambient temperature (°C)` | Referencia exterior |
| `Power (kW)` | Normalizar temperatura por carga |

---

### Código 4500 — Tower resonance | Stop | Operating states

| Campo | Valor |
|---|---|
| Status | **STOP** |
| Ventana de degradación | 2–8 semanas |
| **Lead time recomendado** | **21 días antes del evento** |
| Señal más temprana | La frecuencia de vibración en el acelerómetro se acerca a la frecuencia natural de la torre. Requiere análisis espectral sobre la señal de aceleración. En 10 min la señal es la StdDev y los valores extremos del drive train acceleration. |

**Sensores SCADA necesarios:**

| Columna exacta en CSV | Rol |
|---|---|
| `Drive train acceleration (mm/ss)` | Señal primaria |
| `Tower Acceleration X (mm/ss)` | Aceleración torre eje X |
| `Tower Acceleration y (mm/ss)` | Aceleración torre eje Y |
| `Generator RPM (RPM)` | Velocidad — correlacionar con frecuencia de excitación |
| `Rotor speed (RPM)` | Velocidad de rotor |
| `Wind speed (m/s)` | Velocidad de viento como excitador |
| `Power (kW)` | Carga operativa |

**Feature engineering**: Con datos de 10 min usar los valores Max y StdDev de Tower Acceleration X/Y. Calcular el ratio `Tower Acceleration / Wind speed^2` para normalizar por la presión dinámica del viento.

---

### Código 4510 — Tower oscillation Y level 1 | Stop | Safety stop of WEC

| Campo | Valor |
|---|---|
| Status | **STOP** |
| Ventana de degradación | 1–5 días |
| **Lead time recomendado** | **3 días antes del evento** |
| Señal más temprana | La amplitud de oscilación Y aumenta progresivamente sobre el baseline. El ratio oscilación/velocidad de viento sube antes del stop. |

**Sensores SCADA necesarios:**

| Columna exacta en CSV | Rol |
|---|---|
| `Tower Acceleration y (mm/ss)` | Señal directa |
| `Drive train acceleration (mm/ss)` | Vibración correlacionada |
| `Generator RPM (RPM)` | RPM como excitador |
| `Wind speed (m/s)` | Normalizar amplitud |
| `Wind speed, Standard deviation (m/s)` | Turbulencia |

---

### Código 4520 — Tower oscillation X level 1 | Stop | Safety stop of WEC

| Campo | Valor |
|---|---|
| Status | **STOP** |
| Ventana de degradación | 1–5 días |
| **Lead time recomendado** | **3 días antes del evento** |

**Sensores SCADA necesarios:**

| Columna exacta en CSV | Rol |
|---|---|
| `Tower Acceleration X (mm/ss)` | Señal directa |
| `Drive train acceleration (mm/ss)` | Vibración correlacionada |
| `Generator RPM (RPM)` | RPM como excitador |
| `Wind speed (m/s)` | Normalizar amplitud |

---

### Código 4530 — Tower oscillation Y level 2 | Stop | Safety stop of WEC

| Campo | Valor |
|---|---|
| Status | **STOP** |
| Ventana de degradación | Horas – 1 día |
| **Lead time recomendado** | **6 horas antes del evento** (predecir desde 4510) |
| Señal más temprana | El nivel 2 suele seguir al nivel 1 en horas. Modelar la secuencia 4510 → 4530 como una transición de estado. |

**Sensores SCADA necesarios:** Mismos que 4510.

---

### Código 4540 — Tower oscillation X level 2 | Stop | Safety stop of WEC

| Campo | Valor |
|---|---|
| Status | **STOP** |
| Ventana de degradación | Horas – 1 día |
| **Lead time recomendado** | **6 horas antes del evento** (predecir desde 4520) |

**Sensores SCADA necesarios:** Mismos que 4520.

---

### Código 4600 — PT100 base box temp defect | Warning | Warnings

| Campo | Valor |
|---|---|
| Status | Warning |
| Ventana de degradación | 1–4 días |
| **Lead time recomendado** | **2 días antes del evento** |
| Señal más temprana | La temperatura de base box diverge del valor esperado en función de T ambiente. Aparecen spikes en Max y StdDev antes del fallo del sensor. |

**Sensores SCADA necesarios:**

| Columna exacta en CSV | Rol |
|---|---|
| `Nacelle ambient temperature (°C)` | Referencia exterior |
| `Nacelle ambient temperature, Max (°C)` | Spike detector |
| `Nacelle ambient temperature, StdDev (°C)` | Inestabilidad señal |
| `Ambient temperature (converter) (°C)` | Segunda referencia de temperatura |

---

### Código 4607 — Heating/fan base box faulty | Warning | Warnings

| Campo | Valor |
|---|---|
| Status | Warning |
| Ventana de degradación | 2–6 días |
| **Lead time recomendado** | **4 días antes del evento** |
| Señal más temprana | La temperatura de nacelle no responde al ciclo de calefacción esperado. El delta T nacelle vs ambiente se vuelve anómalo — la nacelle no se calienta lo suficiente en frío. |

**Sensores SCADA necesarios:**

| Columna exacta en CSV | Rol |
|---|---|
| `Nacelle temperature (°C)` | Temperatura interior |
| `Nacelle ambient temperature (°C)` | Temperatura exterior |
| `Nacelle temperature, Standard deviation (°C)` | Irregularidad del ciclo térmico |
| `Nacelle temperature, Max (°C)` | Máximo ciclo |
| `Nacelle temperature, Min (°C)` | Mínimo ciclo |

---

### Código 5510 — Low hydraulic pressure | Stop | Mechanical error

| Campo | Valor |
|---|---|
| Status | **STOP** |
| Ventana de degradación | 2–5 días |
| **Lead time recomendado** | **3 días antes del evento** |
| Señal más temprana | La presión de aceite de caja va cayendo por debajo del baseline esperado para esa temperatura de aceite. El ratio presión/temperatura empieza a deteriorarse días antes. |

**Sensores SCADA necesarios:**

| Columna exacta en CSV | Rol |
|---|---|
| `Gear oil inlet pressure (bar)` | Señal primaria — presión hidráulica |
| `Gear oil inlet pressure, Max (bar)` | Pico de presión |
| `Gear oil inlet pressure, Min (bar)` | Caída mínima — la más diagnóstica |
| `Gear oil pump pressure (bar)` | Presión de la bomba |
| `Gear oil pump pressure, Min (bar)` | Caída de bomba |
| `Gear oil inlet temperature (°C)` | Para normalizar presión por viscosidad |
| `Gear oil temperature (°C)` | Temperatura del aceite |

**Feature engineering**: `presión_normalizada = Gear oil inlet pressure / (1 - k*(Gear oil temperature - T_ref))`. Una presión normalizada decreciente indica fuga o fallo de bomba.

---

### Código 5720 — Brake accumulator defect | Warning | Warnings

| Campo | Valor |
|---|---|
| Status | Warning |
| Ventana de degradación | 1–3 semanas |
| **Lead time recomendado** | **14 días antes del evento** |
| Señal más temprana | La presión del acumulador se recupera más lentamente tras cada evento de frenado. La "firma dinámica" de la presión durante paradas cambia semanas antes del fallo. Requiere analizar la recuperación de presión post-parada. |

**Sensores SCADA necesarios:**

| Columna exacta en CSV | Rol |
|---|---|
| `Gear oil inlet pressure (bar)` | Recuperación de presión post-parada |
| `Gear oil inlet pressure, Min (bar)` | Presión mínima durante frenado |
| `Gear oil pump pressure (bar)` | Estado de la bomba |
| `Generator RPM (RPM)` | Detectar eventos de parada |
| `Power (kW)` | Detectar eventos de parada |

---

### Código 6052 — High yaw motor current | Warning | Warnings

| Campo | Valor |
|---|---|
| Status | Warning |
| Ventana de degradación | 3–10 días |
| **Lead time recomendado** | **7 días antes del evento** |
| Señal más temprana | La corriente del motor yaw para la misma velocidad de viento y mismo cambio de orientación es creciente. La resistencia mecánica del sistema de orientación (engranaje, rodamiento, freno yaw) aumenta progresivamente. |

**Sensores SCADA necesarios:**

| Columna exacta en CSV | Rol |
|---|---|
| `Nacelle position (°)` | Cambios de orientación |
| `Nacelle position, Standard deviation (°)` | Frecuencia de movimiento yaw |
| `Wind direction (°)` | Dirección del viento — demanda de yaw |
| `Vane position 1+2 (°)` | Error de alineación nacelle-viento |
| `Wind speed (m/s)` | Normalizar demanda yaw |

**Nota**: El dataset Kelmarsh no incluye directamente corriente del motor yaw, pero el esfuerzo se puede inferir de la frecuencia de movimientos de nacelle y el error de alineación.

---

### Código 6054 — Easy yaw | Warning | Warnings

| Campo | Valor |
|---|---|
| Status | Warning |
| Ventana de degradación | 1–4 semanas |
| **Lead time recomendado** | **14 días antes del evento** |
| Señal más temprana | La nacelle no sigue el viento con la precisión esperada. El error de alineación nacelle-viento aumenta sostenidamente → pérdida de potencia → modo easy yaw activado más frecuentemente. |

**Sensores SCADA necesarios:**

| Columna exacta en CSV | Rol |
|---|---|
| `Nacelle position (°)` | Orientación real de la nacelle |
| `Wind direction (°)` | Dirección del viento |
| `Vane position 1+2 (°)` | Error de alineación directo |
| `Vane position 1+2, StdDev (°)` | Inestabilidad de alineación |
| `Power (kW)` | Pérdida de producción por desalineación |
| `Wind speed (m/s)` | Variable de control |

**Feature engineering**: `yaw_error = abs(Nacelle position - Wind direction)`. Un yaw error medio creciente (en ventana de 24h) es la señal más limpia para 6054.

---

### Código 6120 — Uncontrolled yaw movement | Stop | Mechanical error

| Campo | Valor |
|---|---|
| Status | **STOP** |
| Ventana de degradación | Horas – 2 días |
| **Lead time recomendado** | **12 horas antes del evento** (predecir desde 6052/6054) |
| Señal más temprana | La nacelle cambia de posición sin comando. Suele seguir a eventos previos de 6052 o 6054. Verificar también cable windings y posición de vane. |

**Sensores SCADA necesarios:**

| Columna exacta en CSV | Rol |
|---|---|
| `Nacelle position (°)` | Movimiento sin comando |
| `Nacelle position, Standard deviation (°)` | Varianza de posición anómala |
| `Cable windings from calibration point` | Estado del cable |
| `Vane position 1+2 (°)` | Señal de vane como referencia |
| `Wind direction (°)` | Comparar con posición nacelle |

---

### Código 6515 — 4-20mA anemometer 1 | Warning | Warnings

| Campo | Valor |
|---|---|
| Status | Warning |
| Ventana de degradación | 1–5 días |
| **Lead time recomendado** | **3 días antes del evento** |
| Señal más temprana | La señal del anemómetro 1 empieza a tener caídas fuera del rango 4–20 mA (clipping o dropout) antes del fallo completo. La curva potencia-viento se vuelve ruidosa o inconsistente. |

**Sensores SCADA necesarios:**

| Columna exacta en CSV | Rol |
|---|---|
| `Wind speed Sensor 1 (m/s)` | Señal del anemómetro 1 directamente |
| `Wind speed Sensor 1, Standard deviation (m/s)` | Inestabilidad de señal |
| `Wind speed Sensor 1, Minimum (m/s)` | Detectar dropouts a cero |
| `Wind speed Sensor 1, Maximum (m/s)` | Detectar clipping |
| `Wind speed Sensor 2 (m/s)` | Referencia comparación |
| `Power (kW)` | Coherencia velocidad-potencia |

**Feature engineering**: `abs(Wind speed Sensor 1 - Wind speed Sensor 2)`. Divergencia entre sensores es señal directa.

---

### Código 6525 — 4-20mA anemometer 2 | Warning | Warnings

| Campo | Valor |
|---|---|
| Status | Warning |
| Ventana de degradación | 1–5 días |
| **Lead time recomendado** | **3 días antes del evento** |

**Sensores SCADA necesarios:**

| Columna exacta en CSV | Rol |
|---|---|
| `Wind speed Sensor 2 (m/s)` | Señal del anemómetro 2 |
| `Wind speed Sensor 2, Standard deviation (m/s)` | Inestabilidad |
| `Wind speed Sensor 2, Minimum (m/s)` | Dropouts |
| `Wind speed Sensor 2, Maximum (m/s)` | Clipping |
| `Wind speed Sensor 1 (m/s)` | Referencia comparación |
| `Power (kW)` | Coherencia velocidad-potencia |

---

### Código 6530 — Anemometer defect | Stop | Sensor error

| Campo | Valor |
|---|---|
| Status | **STOP** |
| Ventana de degradación | 2–6 días |
| **Lead time recomendado** | **4 días antes del evento** |
| Señal más temprana | Divergencia creciente entre anemómetros 1 y 2. La relación potencia/viento queda fuera de la curva de potencia esperada. Si 6515 o 6525 ya han disparado, este fallo es el siguiente paso. |

**Sensores SCADA necesarios:**

| Columna exacta en CSV | Rol |
|---|---|
| `Wind speed (m/s)` | Media de ambos sensores |
| `Wind speed Sensor 1 (m/s)` | Sensor 1 individual |
| `Wind speed Sensor 2 (m/s)` | Sensor 2 individual |
| `Wind speed, Standard deviation (m/s)` | Inestabilidad de la señal combinada |
| `Power (kW)` | Coherencia curva de potencia |
| `Vane position 1+2 (°)` | Orientación — confirmar que el viento es real |

---

### Código 6620 — Vane defect | Stop | Sensor error

| Campo | Valor |
|---|---|
| Status | **STOP** |
| Ventana de degradación | 2–6 días |
| **Lead time recomendado** | **4 días antes del evento** |
| Señal más temprana | La dirección indicada por el vane diverge de la posición de nacelle esperada. La StdDev de la dirección de viento sube anormalmente. |

**Sensores SCADA necesarios:**

| Columna exacta en CSV | Rol |
|---|---|
| `Wind direction (°)` | Señal del vane (dirección viento) |
| `Wind direction, Standard deviation (°)` | Inestabilidad de la señal |
| `Wind direction, Minimum (°)` | Detectar dropouts |
| `Wind direction, Maximum (°)` | Detectar spikes |
| `Nacelle position (°)` | Referencia de orientación |
| `Vane position 1+2 (°)` | Posición combinada vane |
| `Vane position 1+2, StdDev (°)` | Inestabilidad vane |

---

### Código 6622 — Vane 2 defect | Warning | Warnings

| Campo | Valor |
|---|---|
| Status | Warning |
| Ventana de degradación | 2–5 días |
| **Lead time recomendado** | **3 días antes del evento** |

**Sensores SCADA necesarios:**

| Columna exacta en CSV | Rol |
|---|---|
| `Vane position 1+2 (°)` | Señal combinada — buscar divergencia |
| `Vane position 1+2, StdDev (°)` | Inestabilidad |
| `Vane position 1+2, Max (°)` | Spikes |
| `Vane position 1+2, Min (°)` | Dropouts |
| `Nacelle position (°)` | Referencia de orientación real |
| `Wind direction (°)` | Comparar con vane |

---

### Código 6635 — 4-20mA vane 2 | Warning | Sensor error

| Campo | Valor |
|---|---|
| Status | Warning |
| Ventana de degradación | 1–4 días |
| **Lead time recomendado** | **2 días antes del evento** |
| Señal más temprana | La señal 4-20mA del vane 2 empieza a salirse del rango antes del fallo completo. Patrón similar a 6515/6525: StdDev sube, aparecen valores fuera de rango. |

**Sensores SCADA necesarios:**

| Columna exacta en CSV | Rol |
|---|---|
| `Vane position 1+2 (°)` | Señal del vane |
| `Vane position 1+2, StdDev (°)` | Inestabilidad — sube antes del fallo |
| `Vane position 1+2, Max (°)` | Detectar clipping/spikes |
| `Vane position 1+2, Min (°)` | Detectar dropouts |
| `Nacelle position (°)` | Referencia |

---

## Resumen consolidado: columnas a extraer del dataset

Las siguientes 60 columnas (de las 303 totales) cubren el 100% de los fallos listados. El resto son métricas de disponibilidad, contadores de energía y señales derivadas prescindibles para el modelo predictivo.

### Grupo 1 — Viento y orientación (crítico para normalización)
- `Wind speed (m/s)`
- `Wind speed, Standard deviation (m/s)`
- `Wind speed Sensor 1 (m/s)`
- `Wind speed Sensor 1, Standard deviation (m/s)`
- `Wind speed Sensor 1, Minimum (m/s)`
- `Wind speed Sensor 1, Maximum (m/s)`
- `Wind speed Sensor 2 (m/s)`
- `Wind speed Sensor 2, Standard deviation (m/s)`
- `Wind speed Sensor 2, Minimum (m/s)`
- `Wind speed Sensor 2, Maximum (m/s)`
- `Wind direction (°)`
- `Wind direction, Standard deviation (°)`
- `Wind direction, Minimum (°)`
- `Wind direction, Maximum (°)`
- `Nacelle position (°)`
- `Nacelle position, Standard deviation (°)`
- `Vane position 1+2 (°)`
- `Vane position 1+2, StdDev (°)`
- `Vane position 1+2, Max (°)`
- `Vane position 1+2, Min (°)`

### Grupo 2 — Potencia y electricidad
- `Power (kW)`
- `Power (kW)` → usar también `Power, Standard deviation (kW)`, `Power, Minimum (kW)`, `Power, Maximum (kW)`
- `Power factor (cosphi)`
- `Power factor (cosphi), Standard deviation`
- `Reactive power (kvar)`
- `Grid voltage (V)`
- `Grid frequency (Hz)`
- `Current L1 / U (A)`
- `Current L2 / V (A)`
- `Current L3 / W (A)`

### Grupo 3 — Pitch / blades
- `Blade angle (pitch position) A (°)`
- `Blade angle (pitch position) A, Max (°)`
- `Blade angle (pitch position) A, Min (°)`
- `Blade angle (pitch position) A, Standard deviation (°)`
- `Blade angle (pitch position) B (°)`
- `Blade angle (pitch position) B, Max (°)`
- `Blade angle (pitch position) B, Standard deviation (°)`
- `Blade angle (pitch position) C (°)`
- `Blade angle (pitch position) C, Max (°)`
- `Blade angle (pitch position) C, Standard deviation (°)`
- `Motor current axis 1 (A)`
- `Motor current axis 1, Max (A)`
- `Motor current axis 1, StdDev (A)`
- `Motor current axis 2 (A)`
- `Motor current axis 2, Max (A)`
- `Motor current axis 2, StdDev (A)`
- `Motor current axis 3 (A)`
- `Motor current axis 3, Max (A)`
- `Motor current axis 3, StdDev (A)`
- `Temperature motor axis 1 (°C)`
- `Temperature motor axis 2 (°C)`
- `Temperature motor axis 3 (°C)`

### Grupo 4 — Tren de transmisión y caja
- `Drive train acceleration (mm/ss)`
- `Tower Acceleration X (mm/ss)`
- `Tower Acceleration y (mm/ss)`
- `Generator RPM (RPM)`
- `Generator RPM, Max (RPM)`
- `Generator RPM, Min (RPM)`
- `Generator RPM, Standard deviation (RPM)`
- `Rotor speed (RPM)`
- `Gearbox speed (RPM)`
- `Gear oil temperature (°C)`
- `Gear oil temperature, Max (°C)`
- `Gear oil inlet temperature (°C)`
- `Gear oil inlet pressure (bar)`
- `Gear oil inlet pressure, Min (bar)`
- `Gear oil pump pressure (bar)`
- `Metal particle count`
- `Metal particle count counter`
- `Front bearing temperature (°C)`
- `Rear bearing temperature (°C)`

### Grupo 5 — Generador y convertidor
- `Generator bearing front temperature (°C)`
- `Generator bearing front temperature, Max (°C)`
- `Generator bearing rear temperature (°C)`
- `Generator bearing rear temperature, Max (°C)`
- `Ambient temperature (converter) (°C)`
- `Ambient temperature (converter), Max (°C)`
- `Ambient temperature (converter), Min (°C)`
- `Ambient temperature (converter), StdDev (°C)`

### Grupo 6 — Temperaturas nacelle y transformador
- `Nacelle temperature (°C)`
- `Nacelle temperature, Max (°C)`
- `Nacelle temperature, Min (°C)`
- `Nacelle temperature, Standard deviation (°C)`
- `Nacelle ambient temperature (°C)`
- `Nacelle ambient temperature, Max (°C)`
- `Nacelle ambient temperature, StdDev (°C)`
- `Transformer temperature (°C)`
- `Transformer temperature, Max (°C)`
- `Transformer cell temperature (°C)`

### Grupo 7 — Cable y yaw
- `Cable windings from calibration point`
- `Cable windings from calibration point, Max`
- `Cable windings from calibration point, Min`

### Columna de tiempo (obligatoria)
- `Date and time` — resolución 10 minutos, usar como índice temporal

---

## Recomendaciones para el etiquetado del dataset

1. **Columna target**: crear `time_to_fault` (minutos hasta el próximo evento del código en cuestión). Usar regresión, no clasificación binaria.
2. **Ventana negativa**: etiquetar como estado normal los períodos con más de `2 × lead_time` sin ningún fallo del mismo código.
3. **Exclusión de post-fallo**: excluir los 6 primeros intervalos (1 hora) tras un evento — el sistema puede estar en recuperación y contamina el entrenamiento.
4. **Normalización por condición operativa**: para señales térmicas y eléctricas, calcular residuos respecto a un modelo de comportamiento normal (curva de potencia, modelo térmico lineal por power bin).
5. **Código 1070** (drive train): incluir siempre el `Metal particle count` y el `Drive train acceleration` — son las dos señales con mayor lead time y menor ruido del dataset completo.

