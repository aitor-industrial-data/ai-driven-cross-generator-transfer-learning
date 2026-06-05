# Decisiones Técnicas del Pipeline de Mantenimiento Predictivo
## Kelmarsh Wind Farm — Turbina T1 · 2018–2021

---

## Contexto

Este documento registra las decisiones de diseño relevantes tomadas durante el desarrollo del pipeline de mantenimiento predictivo para la turbina 1 de Kelmarsh. El objetivo es que cualquier persona que trabaje sobre este código entienda **por qué** las cosas están como están, no solo **cómo** funcionan.

---

## 1. Selección de Familias de Fallos

### Familias entrenables

A partir del catálogo de códigos de fallo (`fault_log.csv`), se identificaron 7 grupos candidatos. Cuatro se descartaron:

| Familia descartada | Motivo |
|--------------------|--------|
| `sensors_anemometer` | Todos los eventos se concentran en un único mes (diciembre 2021). Sin distribución temporal suficiente para aprender. |
| `tower_vibration` | Los fallos de vibración de torre ocurren en escala de segundos. La resolución de SCADA (10 minutos) es incompatible con la dinámica del fallo. |
| `drivetrain` | Un único evento histórico en todo el dataset. Imposible entrenar. |

Las cuatro familias entrenables son:

| Familia | Eventos | Tipo de degradación |
|---------|---------|---------------------|
| `yaw_cable` | 187 | Acumulación gradual — contador de vueltas de cable sube linealmente |
| `brake_hydro` | 58 | Degradación mecánica/hidráulica — presión y temperatura |
| `generator` | 55 | Degradación térmica — temperatura de rodamientos |
| `pitch_bat` | 30 | Degradación electroquímica — baterías bajo estrés térmico |

---

## 2. Lead Times

### Primera iteración (descartada)

Los lead times iniciales eran: `yaw_cable=168h, brake_hydro=120h, generator=120h, pitch_bat=336h`.

Con `yaw_cable=168h`, el 47% del dataset quedaba etiquetado como positivo. Con ese nivel de desbalance, el modelo no necesita discriminar: clasificar todo como positivo da un Recall=1.0 trivialmente. El AUC-ROC resultante era 0.51 (azar).

### Lead times definitivos

| Familia | Lead time | Criterio |
|---------|-----------|---------|
| `yaw_cable` | **72h** | Con 72h el porcentaje de positivos baja al 25%, nivel donde el modelo aprende a discriminar. La señal de `cable_windings` es claramente visible 3 días antes. |
| `brake_hydro` | **120h** | Sin cambio. La degradación del acumulador y la presión de la bomba son visibles 5 días antes. |
| `generator` | **120h** | Sin cambio. El calentamiento de rodamientos sube gradualmente en los 5 días previos. |
| `pitch_bat` | **336h** | Sin cambio. La degradación de baterías bajo estrés térmico es un proceso lento, detectable hasta 2 semanas antes. |

---

## 3. Feature Engineering

### Por qué no se usan features de pendiente (slope)

Durante el desarrollo se probaron features de pendiente lineal (`slope`) para ventanas de 1h, 6h, 24h y 7 días. El análisis de AUC univariante mostró AUC = 0.500 en todas las familias y todas las ventanas — equivalente a predicción aleatoria.

La razón es física: en turbinas eólicas los sensores fluctúan continuamente con el viento, la carga y la temperatura exterior. La degradación no produce una pendiente monótona, sino **valores extremos cada vez más frecuentes**. La señal está en la frecuencia de excedencia de umbrales históricos, no en la pendiente puntual.

### Features implementadas

Para cada sensor, en cuatro ventanas (1h, 6h, 24h, 7d):
- `mean`: nivel promedio — detecta derivas lentas
- `std`: variabilidad — detecta inestabilidad creciente
- `p95`: percentil 95 — el sensor empieza a tocar valores extremos
- `exceedance`: fracción del tiempo por encima del p90 del baseline — frecuencia de excedencia

Más un `baseline_ratio` en ventana 7d: media actual dividida entre media de los primeros 180 días de operación limpia.

### Decisión sobre `pitch_bat`: temperatura absoluta vs delta

En la primera implementación, `nacelle_ambient_temperature_c` era la feature más importante para `pitch_bat`. El modelo estaba aprendiendo **estacionalidad** (en invierno hay más fallos de batería) en lugar de **degradación** (la batería no aguanta el frío).

La solución fue sustituir las temperaturas absolutas de los motores por deltas motor-ambiente (`t_motorX_vs_ambient`). Con esta feature, el modelo aprende «el motor no está calentando lo esperado respecto al frío exterior», que es la señal real de batería degradada.

---

## 4. Modelo y Entrenamiento

### LightGBM con class_weight='balanced'

Se eligió LightGBM por tres motivos:
1. Maneja bien el desequilibrio de clases con `class_weight='balanced'`
2. Tolerante a features correlacionadas (natural en sensores de una misma turbina)
3. Early stopping integrado — el número de árboles se determina automáticamente

### Split temporal, no aleatorio

El split train/test es **temporal**: 80% más antiguo para train, 20% más reciente para test. Un split aleatorio permitiría que el modelo vea en train datos de 2021 y prediga en test datos de 2018, introduciendo data leakage temporal.

### Selección de threshold

El threshold de clasificación no es fijo (0.5) sino adaptativo por familia: se elige el menor valor de probabilidad que mantiene una Precision por fila ≥ 10%. El objetivo es maximizar el Recall (detectar fallos) con un mínimo de calidad en las alertas.

---

## 5. Métrica Principal: Event Recall

Las métricas por fila (Precision/Recall/F1 calculados sobre cada intervalo de 10 minutos) son útiles para el desarrollo pero no reflejan el valor operativo del sistema.

Lo que importa en producción es: **de los N fallos reales que ocurrieron en el período de test, ¿cuántos recibieron al menos una alerta en las horas previas?**

Esta métrica se denomina **Event Recall** en el código. Un Event Recall de 0.90 significa que el sistema habría dado aviso antes del 90% de los fallos reales.

### Resultados finales (período de test: 2021-03 → 2021-12)

| Familia | Fallos en test | Detectados | Event Recall |
|---------|---------------|------------|-------------|
| `yaw_cable` | 24 | 24 | **1.00** ✅ |
| `generator` | 4 | 4 | **1.00** ✅ |
| `pitch_bat` | 2 | 2 | **1.00** ✅ |
| `brake_hydro` | 3 | 1 | **0.33** — ver nota |

---

## 6. Limitación Documentada: brake_hydro en Test

Los 3 fallos de `brake_hydro` en el período de test son:

| Fecha | Código | Descripción | Tipo |
|-------|--------|-------------|------|
| 2021-07-25 | 5510 | Low hydraulic pressure | Rotura abrupta de presión — no predecible |
| 2021-08-12 | 5720 | Brake accumulator defect | Fallo en cascada provocado por `3000 Frequency converter not ready` — causa raíz en otro sistema |
| 2021-09-06 | 2125 | Timeout brake closed | Fallo en cascada junto con `2674 Overload generator heating` — causa raíz térmica |

Ninguno de los tres es un fallo de degradación gradual del sistema hidráulico. El Event Recall de 0.33 en test es el **techo físico** para este conjunto de fallos específicos, no una limitación del modelo. El modelo sí aprende la degradación hidráulica gradual (visible en el período de train), pero el período de test no contiene ese tipo de eventos.

---

## 7. Estructura de Archivos

```
data/
  bronze/          ← archivos CSV anuales originales (sin modificar)
  silver/
    turbine_1_telemetry_clean.parquet   ← producido por 03
    fault_targets_grouped.parquet       ← producido por 03
    dataset_labeled.parquet             ← producido por 04
    features_{familia}.parquet          ← producidos por 05 (uno por familia)
  models/
    model_{familia}.pkl                 ← modelos serializados, producidos por 06
    results_{familia}.json              ← métricas de entrenamiento
    feature_importance_{familia}.png    ← gráficos de importancia
    pr_curve_{familia}.png              ← curvas Precision-Recall
    timeline_{familia}.png              ← timelines de alertas vs fallos reales
```

---

## 8. Siguiente Fase (fuera del alcance actual)

La siguiente fase del proyecto es el transfer learning a turbinas nuevas: entrenar un modelo base con el histórico de T1 y reentrenarlo con los primeros meses de datos de T2, T3 y T4 para que no empiece desde cero. Esta fase está pendiente hasta validar el rendimiento del modelo base en T1.


---

## 9. Cambio de Clasificación a Regresión

### El problema de Precision con clasificación binaria

Con el enfoque de clasificación binaria (`is_pre_fault`), la Precision máxima alcanzable
con Recall ≥ 0.80 estaba matemáticamente limitada por la densidad de fallos en el dataset.

Por ejemplo, con `generator` en el período de test (9 meses, 4 fallos, lead time 120h):
- Filas positivas reales: 4 fallos × 120h × 6 pasos/h = 2.880 filas
- Total filas en test: ~42.000 filas
- Precision máxima teórica ≈ 2.880 / 42.000 ≈ 0.07

No existe threshold ni feature engineering que supere ese límite con un único modelo
de clasificación binaria sobre intervalos de 10 minutos.

### La solución: regresión de `hours_to_fault`

En lugar de "¿hay un fallo en las próximas N horas?", el modelo resuelve
"¿cuántas horas faltan hasta el próximo fallo?"

**Ventajas:**
- La métrica principal pasa a ser MAE en horas — directamente interpretable
- "El modelo predice el fallo del generador con un error medio de ±18h" es mucho
  más informativo que "Precision=0.10, Recall=0.91"
- El umbral de alerta se define en horas (ej: "alerta si pred < 48h"),
  con significado operativo directo
- Precision y Recall se calculan sobre ese umbral de horas y son sustancialmente
  más altas que con clasificación binaria

**Umbral de alerta por familia:**

| Familia | Lead time | Umbral de alerta |
|---------|-----------|-----------------|
| `yaw_cable` | 72h | 48h |
| `generator` | 120h | 72h |
| `brake_hydro` | 120h | 72h |
| `pitch_bat` | 336h | 168h |

El umbral de alerta es conservador respecto al lead time para garantizar
margen de reacción del equipo de mantenimiento.
