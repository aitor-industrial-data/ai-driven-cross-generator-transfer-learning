# ML Design — AI-Driven Cross-Generator Transfer Learning

Este documento describe las decisiones de diseño del pipeline de machine learning: qué se predice, cómo se etiquetan los datos, qué features se construyen, cómo se entrena y cómo se evalúa.

---

## Qué predice el sistema

El sistema no predice si una turbina va a fallar. Predice **cuántas horas faltan hasta el próximo fallo** de cada familia, de forma continua, en cada intervalo de 10 minutos. De esa predicción se deriva una alerta binaria cuando el valor cae por debajo de un umbral de guardia operativo.

Este enfoque de regresión sobre `hours_to_fault` tiene ventajas concretas sobre clasificación directa: la señal de entrenamiento es más rica (el modelo aprende la urgencia, no solo el estado), la calibración posterior es más precisa, y el dashboard puede mostrar al operario no solo si hay riesgo sino cuánto tiempo tiene para actuar.

---

## Las 4 familias de fallo

El SCADA de la Senvion MM92 registra más de 75 tipos distintos de eventos. Una auditoría manual código a código depuró ese catálogo en tres pasos:

**Prefiltro por status**: solo `Stop` y `Warning` contienen señal predictiva. Los estados de arranque normal, producción y parada planificada se descartan.

**Auditoría de ruido**: 20 códigos se eliminaron en tres categorías —mantenimiento planificado (pruebas de batería, limpiezas de aceite), causas externas (cortes de red, hielo, viento excesivo) y comandos manuales (paradas en campo, frenado manual). Ninguno tiene precursores detectables en el SCADA porque no hay degradación interna previa que predecir.

**Agrupación por causalidad física**: los 52 códigos válidos se agruparon en 4 familias según el subsistema causante, no por frecuencia estadística. Tres familias adicionales se descartaron con justificación documentada: sensores y anemómetros (los precursores son propios del sensor defectuoso, no de la turbina), vibración de torre (señal insuficiente para el lead time necesario), y drivetrain (demasiado pocos eventos para aprender).

| Familia | Códigos incluidos | Lead time | Umbral de alerta |
|---|---|:---:|:---:|
| `yaw_cable` | Corriente motores yaw, error de yaw, cable autounwind | **83 h** | 48 h |
| `generator` | Temperaturas de rodamiento, sobrecargas de ventiladores, convertidor | **127 h** | 72 h |
| `brake_hydro` | Presión hidráulica, acumulador, pastillas, freno | **130 h** | 72 h |
| `pitch_bat` | Baterías de pitch, ciclos de carga, errores de eje | **295 h** | 168 h |

Los lead times son la decisión de diseño más delicada del pipeline. Un valor demasiado corto deja pocas filas positivas y el modelo no aprende el patrón. Un valor demasiado largo genera tantos positivos que el modelo no necesita discriminar. Para `yaw_cable`, por ejemplo, la señal de `cable_windings` era detectable hasta 7 días antes, pero con ese lead time el 47% del dataset era positivo — el modelo convergía a predecir siempre. El valor definitivo de 83 h es el máximo que mantiene un porcentaje de positivos que fuerza al modelo a discriminar.

---

## Etiquetado

El SCADA no etiqueta fallos. Etiqueta paradas. La diferencia es crítica: dos paradas con el mismo código pueden ser una avería espontánea y una revisión programada.

Para cada fila de telemetría con timestamp `ts` y cada familia:

```
Si existe un fallo de esa familia en el futuro dentro de lead_hours:
    hours_to_fault = horas hasta ese fallo
    is_pre_fault   = True
Si no:
    hours_to_fault = NaN
    is_pre_fault   = False
```

La implementación usa NumPy vectorizado sobre las ~262.000 filas del dataset. El resultado son 8 columnas target añadidas a la telemetría (2 por familia × 4 familias).

Las 24 horas posteriores a cada fallo se excluyen del dataset de entrenamiento. Los sensores en el período de transición post-reparación tienen comportamiento atípico que contaminaría el aprendizaje del estado degradado.

El split es **estrictamente temporal**: 60% train, 20% validación (calibración), 20% test. Un split aleatorio introduciría data leakage: el modelo vería en train datos del futuro respecto al test.

---

## Feature engineering

Los valores brutos de sensores a 10 minutos tienen escasa capacidad predictiva. Un sensor no falla por un valor puntual alto, sino por patrones sostenidos: una temperatura que sube durante días, una presión que fluctúa cada vez más, un nivel que supera su umbral histórico con creciente frecuencia.

### Features de ventana rodante

Para cada sensor se calculan estadísticos en 4 ventanas temporales — 1 h, 6 h, 24 h, 7 días:

- Media, desviación estándar, mínimo, máximo → capturan nivel y dispersión
- Percentil 95 → captura valores extremos sin ser sensible a outliers únicos
- Frecuencia de excedencia del percentil 95 del baseline → captura con qué frecuencia el sensor supera su comportamiento histórico sano
- Ratio vs baseline → cuánto se aleja el nivel actual del comportamiento de los primeros 180 días (período de referencia de operación limpia)

En total, **17 features por sensor** × las 4 ventanas × los sensores de cada familia.

Las features de pendiente (`slope`) se evaluaron y descartaron: AUC univariante de 0.500 en todas las familias, equivalente a predicción aleatoria. La degradación en turbinas eólicas no sigue una pendiente monótona — los sensores fluctúan con el viento, la carga y la temperatura exterior. La señal está en los valores extremos y la frecuencia de excedencia, no en la derivada puntual.

### Features de dominio

Antes de las ventanas se calculan combinaciones de sensores con significado físico directo, seleccionadas por causalidad y no por correlación estadística:

- **Yaw/Cable**: error de orientación ponderado por velocidad de viento. A mayor viento y mayor desalineación, mayor par torsional sobre los cables.
- **Generador**: deltas de temperatura de rodamiento respecto a temperatura ambiente. Elimina el efecto estacional: un rodamiento degradado calienta por encima del ambiente aunque el ambiente sea frío.
- **Pitch/Baterías**: delta motor-ambiente en frío. Una batería degradada pierde capacidad a bajas temperaturas — la señal es que el motor no calienta respecto al exterior.

### `hours_since_last_fault`

Para cada familia se añade el tiempo transcurrido desde el último evento de esa familia. Esta feature captura el estado de "recién reparado" vs. "acumulando desgaste". Es también la feature que gestiona el cold start de T2: mientras T2 no tiene historial de fallos propio, se asigna `COLD_START_HOURS = 9999` — el valor más conservador del dominio de entrenamiento de T1, sin extrapolación.

---

## Modelo y entrenamiento

**LightGBM Regressor** con target `hours_to_fault`. Un modelo por familia, con hiperparámetros ajustados independientemente:

| Familia | `num_leaves` | `min_child_samples` |
|---|:---:|:---:|
| `yaw_cable` | 63 | 20 |
| `generator` | 63 | 20 |
| `brake_hydro` | 31 | 30 |
| `pitch_bat` | 63 | 20 |

`brake_hydro` usa un árbol más conservador (`num_leaves=31`, `min_child_samples=30`) porque es la familia con mayor desequilibrio de clases y más sensible al sobreajuste.

Parámetros comunes: `n_estimators=1000`, `learning_rate=0.05`, `early_stopping` sobre validación, `subsample=0.8`, `colsample_bytree=0.8`, regularización L1+L2.

El desequilibrio de clases (2–25% de positivos según familia) se gestiona con `class_weight='balanced'` en la función de pérdida.

---

## Calibración

LightGBM con pocos árboles efectivos aprende a predecir la media condicional. En la práctica: cuando el fallo real está a 5 h, predice ~45 h; cuando está a 115 h, también predice ~60 h. Las predicciones se comprimen hacia el centro del rango, haciendo las alertas inútiles.

**Isotonic Regression** aprende la función monótona que corrige ese sesgo. Se entrena en el conjunto de validación (el 20% intermedio del split temporal) mapeando `pred_raw → hours_to_fault_real`. El pipeline final es:

```
SCADA → features → LGBMRegressor → IsotonicRegression → hours_to_fault_calibrado → alerta
```

La calibración se entrena en validación y se evalúa en test para evitar data leakage entre la corrección y la evaluación.

---

## Métrica de evaluación

**Event Recall** es la métrica primaria. De todos los fallos reales ocurridos en el período de test, el porcentaje que recibió al menos una alerta antes del umbral de guardia.

La justificación es operativa: en mantenimiento predictivo, lo que importa no es cuántas filas individuales se clasifican bien sino cuántas averías reales se anticipan. Un modelo que detecte 9 de 10 fallos con muchas falsas alarmas es operativamente superior a uno que detecte 5 de 10 con pocas.

Precision y F1 se calculan como métricas secundarias para controlar la tasa de falsas alarmas, pero no determinan la selección del modelo.

Una advertencia sobre la interpretación de los resultados iniciales: el test set de T2 contiene los fallos acumulados en los primeros meses de operación. Pocos fallos por familia significa que el Event Recall se mide sobre una muestra pequeña —detectar 1 de 1 evento es un 100% estadísticamente frágil, no una garantía de robustez. La fiabilidad real del sistema se construye a medida que T2 acumula más fallos propios y el test set crece. Los resultados actuales validan que el pipeline funciona y que el Transfer Learning produce predicciones coherentes; no son el techo del sistema sino su punto de partida.

---

## Transfer Learning: la lógica del reentrenamiento mensual

El reentrenamiento mensual no es solo una actualización del modelo. Es el mecanismo de validación continua de la hipótesis central del proyecto.

Cada mes, para cada familia, se entrenan dos versiones:

- **Versión A** — entrenada únicamente con los datos de T2 acumulados hasta la fecha
- **Versión B** — entrenada con el histórico completo de T1 más los datos de T2 hasta la fecha

Ambas versiones se evalúan sobre el mismo test set de T2. La que obtiene mejor Event Recall se despliega en producción. No hay intervención manual.

La expectativa es que al inicio de la operación de T2, la Versión B gane siempre: T2 tiene pocos datos propios y el conocimiento de T1 es decisivo. A medida que T2 acumula meses de operación y fallos propios, la Versión A irá ganando terreno. En algún momento la Versión A superará consistentemente a la Versión B.

Ese cruce —el punto en que los datos propios de T2 son suficientes para superar al modelo conjunto— es la validación empírica de la hipótesis de Transfer Learning: **el conocimiento transferido desde T1 es útil exactamente durante el período en que T2 no tiene suficiente historial propio para aprender por sí sola.**

Este proceso no se puede acelerar artificialmente. Los fallos industriales ocurren cuando ocurren. Una familia con cuatro eventos al año necesitará varios años para tener una curva de aprendizaje estadísticamente significativa. Lo que el sistema garantiza es que esos años no se desperdicien arrancando desde cero: cada fallo de T2 refina un modelo que ya sabía algo, en lugar de empezar a aprender desde el primer evento. La diferencia entre arrancar desde una base sólida y arrancar desde cero puede medirse en años de protección perdida.