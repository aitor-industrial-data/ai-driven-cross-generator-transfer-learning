# Production — AI-Driven Cross-Generator Transfer Learning

Este documento describe el sistema en producción: qué ejecuta cada componente, en qué orden, con qué lógica de decisión, y cómo se gestiona el cold start y la evolución del modelo en el tiempo.

---

## Visión general del ciclo de vida

El sistema tiene dos ritmos de ejecución independientes, disparados por EventBridge:

```
Cada noche
  └─ Lambda t2-inference-lambda
       → actualiza Feature Stores de T2
       → produce predicciones para el dashboard

Día 1 de cada mes
  └─ ECS Fargate t2-retrain
       → actualiza el fault log de T2
       → reentrena dos versiones del modelo por familia
       → despliega el ganador automáticamente
```

Todo el estado del sistema —telemetría, modelos, predicciones, logs— vive en S3. Ningún componente mantiene estado local entre ejecuciones.

---

## Inferencia diaria (`t2_daily_inference_serverless.py`)

La Lambda ejecuta 7 pasos en secuencia cada noche.

**Paso 1 — Telemetría Bronze**
Descarga el Parquet de telemetría de T2 desde `bronze/turbine_2_telemetry_clean.parquet/` y filtra por `timestamp <= ahora`. Esto garantiza que una ejecución nunca consume datos del futuro, aunque el archivo Bronze contenga registros simulados hasta 2030.

**Paso 2 — Baseline de T2**
Carga `models/turbine_2_baseline.json` con la media y el percentil 90 por sensor durante los primeros 180 días de operación de T2 (estado sano de referencia). Si no existe, las features de excedencia y ratio-vs-baseline se calculan con diccionarios vacíos —degradación suave, no un fallo crítico.

**Paso 3 — Fault log de T2**
Carga `models/turbine_2_fault_log.csv`. Si no existe —T2 sin ningún fallo registrado aún— se inicializa un DataFrame vacío y el sistema entra en **modo cold start completo**.

**Paso 4 — Ventana de cálculo compartida**
Antes de calcular features por familia, determina el último timestamp almacenado en cada Feature Store. Toma el mínimo entre familias como punto de corte global: si una familia va rezagada, todas recalculan desde ahí. Esto garantiza que los Feature Stores estén sincronizados.

Para las filas nuevas construye una ventana que incluye las 1.008 filas previas (7 días a pasos de 10 minutos) como contexto rolling. Las features de ventana de 7 días necesitan ese contexto para ser válidas en los primeros registros del día.

**Paso 5 — Feature Stores por familia**
Para cada una de las 4 familias, calcula features de dominio y rolling statistics sobre la ventana de cálculo, añade `hours_since_last_{family}` desde el fault log, y hace append al Parquet de esa familia en S3. Los duplicados por timestamp se eliminan con `keep='last'` para que la operación sea idempotente si Lambda se reintenta.

Cold start por familia: si el fault log no contiene ningún evento de esa familia, `hours_since_last_{family}` se fija a `9999.0` y su logaritmo a `log1p(9999)`. Este valor corresponde al extremo superior del dominio de entrenamiento de T1 —sin extrapolación fuera del espacio conocido.

**Paso 6 — Inferencia**
Para cada familia carga el modelo de producción con la siguiente prioridad:
1. `models/t2_model_{family}.pkl` — modelo reentrenado con datos de T2
2. `models/t1_model_{family}.pkl` — modelo de T1, fallback hasta que exista el primero

Toma las últimas 1.008 filas del Feature Store (ventana de 7 días), extrae la última fila como vector de features, y produce:
- `raw_pred`: predicción bruta del LGBMRegressor, recortada a `[0, lead_hours]`
- `cal_pred`: predicción calibrada por IsotonicRegression, recortada al mismo rango
- `is_alert`: `True` si `cal_pred ≤ alert_h` para esa familia

**Paso 7 — Log de predicciones**
Hace append de los resultados del día a `models/t2_predictions_log.csv`, deduplicando por `(last_data_ts, family)`. Si el log existente no se puede leer por cualquier error, la operación falla con excepción —nunca sobreescribe el historial con un archivo roto.

---

## Reentrenamiento mensual (`t2_monthly_retrain.py`)

El contenedor Fargate ejecuta 4 pasos el día 1 de cada mes.

**Paso 0 — Actualizar fault log**
Lee `bronze/turbine_2_status_2026_2030.csv` (el archivo de eventos SCADA de T2), filtra por `timestamp <= hoy`, aplica el mismo pipeline de auditoría de códigos que se usó en los notebooks de T1: descarta los 30 códigos de ruido (mantenimiento planificado, causas externas, operativos humanos), mapea los restantes a las 4 familias usando `FAMILY_CODES`, normaliza los timestamps a intervalos de 10 minutos, y guarda el resultado en `models/turbine_2_fault_log.csv`.

Esta lógica de filtrado es idéntica a la del notebook `01_eda_status_and_events`. El fault log es la fuente de verdad para el etiquetado de T2.

**Paso 1 — Cargar Feature Stores de T1**
Los Feature Stores de T1 (`models/t1_features_{family}.parquet`) se generaron durante el desarrollo en los notebooks y están subidos a S3. Ya contienen las columnas target (`hours_to_fault`, `is_pre_fault`) porque T1 tiene historial completo de averías. No se regeneran en producción.

**Paso 2 — Etiquetar T2 y entrenar dos versiones**
Para cada familia:

*Etiquetado de T2*: aplica la misma lógica de `hours_to_fault` / `is_pre_fault` usada en el notebook `04_labeling`, ahora sobre los Feature Stores de T2 acumulados hasta la fecha. Las 24 horas posteriores a cada fallo se excluyen. El resultado es el Feature Store de T2 etiquetado para esta ronda de entrenamiento.

*Split del test set*: el 20% temporal final del Feature Store de T2 etiquetado se separa como test set **antes de entrenar ninguna versión**. Ambas versiones se evalúan sobre exactamente el mismo test set, garantizando una comparación justa.

*Versión A — Solo T2*: entrena LGBMRegressor + IsotonicRegression únicamente sobre datos de T2. Split temporal 60/20/20.

*Versión B — T1 + T2*: concatena los Feature Stores de T1 y T2 sobre las columnas comunes. Para evitar que el gap de 4 años entre fuentes distorsione el split temporal, T1 y T2 se dividen por separado en train/val y luego se concatenan por tramo. El test set es siempre el de T2, nunca el de T1.

*Selección del ganador*: se comparan por `F1(Event Recall, Precision)` sobre el test set de T2. En caso de empate, gana T1+T2 por tener más datos de entrenamiento. Si una versión no tiene datos suficientes para entrenar (menos de 10 positivos en train o 5 en validación), se descarta y la otra gana automáticamente.

**Paso 3 — Desplegar y registrar**
El modelo ganador se serializa como `models/t2_model_{family}.pkl`. A partir de esa noche, la Lambda de inferencia lo usará en el Paso 6.

Los resultados del reentrenamiento se guardan en dos formatos:
- `models/t2_retrain_results.json` — resultado del último reentrenamiento (lo consume el dashboard)
- `models/t2_retrain_log.csv` — histórico acumulativo de todos los reentrenamientos, con métricas de ambas versiones por familia y fecha

---

## Estructura de S3

```
s3://ai-driven-cross-generator-transfer-learning/
│
├── bronze/
│   ├── turbine_2_telemetry_clean.parquet/   ← SCADA de T2 (particionado)
│   └── turbine_2_status_2026_2030.csv       ← eventos SCADA de T2
│
├── models/
│   ├── t1_features_{family}.parquet    ← Feature Stores de T1 (estáticos, con targets)
│   ├── t2_features_{family}.parquet    ← Feature Stores de T2 (crecen cada noche)
│   ├── t1_model_{family}.pkl           ← modelos T1 (fallback inicial)
│   ├── t2_model_{family}.pkl           ← modelos en producción (actualizados mensualmente)
│   ├── turbine_2_baseline.json         ← media y p90 por sensor (180 días iniciales)
│   ├── turbine_2_fault_log.csv         ← fallos técnicos reales de T2 (actualizado mensual)
│   ├── t2_predictions_log.csv          ← histórico de predicciones diarias
│   ├── t2_retrain_results.json         ← último reentrenamiento (dashboard)
│   └── t2_retrain_log.csv             ← histórico acumulativo de reentrenamientos
│
└── html/
    └── dashboard_t2_v2.html            ← dashboard del operario (acceso público)
```

---

## Cold start y evolución del modelo

Al inicio de la operación de T2, el sistema funciona íntegramente con los modelos de T1 como fallback. A medida que T2 acumula datos:

| Fase | Estado | Modelo en producción |
|---|---|---|
| T2 sin fallos propios | `hours_since_last = 9999`, fault log vacío | `t1_model_{family}.pkl` (fallback) |
| T2 con primeros fallos | Etiquetado posible, pocos positivos | Versión B (T1+T2) gana por más datos |
| T2 con meses de operación | Positivos suficientes en train y test | Competición real entre A y B cada mes |
| T2 con historial maduro | T2 tiene señal propia suficiente | Versión A (solo T2) empieza a ganar |

El log acumulativo `t2_retrain_log.csv` registra qué versión gana cada mes y por qué margen. La evolución de ese dato a lo largo del tiempo es la validación empírica de la hipótesis del proyecto.

Es importante entender qué significa "en producción" en el contexto de mantenimiento predictivo industrial. El sistema está operativo y produce predicciones reales desde el primer día, pero su fiabilidad no es estática: crece con cada fallo nuevo que T2 registra, con cada reentrenamiento mensual que incorpora más señal real, con cada año adicional de operación que enriquece los Feature Stores. Un modelo entrenado con diez fallos de una familia predice distinto a uno entrenado con cuarenta. Llegar a cuarenta fallos puede llevar años en equipos industriales con alta fiabilidad.

La arquitectura está diseñada para este horizonte temporal largo. No es un sistema que se despliega y se olvida: es un sistema que mejora solo, de forma continua y documentada, mientras la turbina opera. El Transfer Learning desde T1 compra tiempo —años de protección que de otro modo serían operación a ciegas— mientras T2 construye su propio historial.
