# Decisions — AI-Driven Cross-Generator Transfer Learning

Registro de las decisiones técnicas relevantes tomadas durante el desarrollo, con la alternativa considerada y la justificación de la elección. Este documento existe porque las decisiones de diseño en ML son tan importantes como el código —y con frecuencia mucho más difíciles de reconstruir a posteriori.

---

## 1. Qué predecir: regresión sobre `hours_to_fault` vs. clasificación binaria

**Alternativa descartada:** clasificación binaria `is_pre_fault ∈ {0, 1}`.

**Decisión:** regresión sobre `hours_to_fault` continuo, con `is_pre_fault` derivado como columna secundaria.

**Justificación:** la clasificación binaria colapsa toda la ventana de pre-fallo en una sola clase. El modelo no aprende que estar a 5 horas del fallo es cualitativamente distinto a estar a 120 horas — ambos son simplemente "1". La regresión preserva esa gradiente: el modelo aprende la urgencia, no solo el estado. Esto produce calibración más precisa, alertas con antelación variable según la familia, y un dashboard que puede mostrar al operario cuánto tiempo tiene para actuar en lugar de solo si hay riesgo.

---

## 2. Auditoría manual del catálogo de fault codes

**Alternativa descartada:** filtrado automático por frecuencia o por categoría contractual SCADA.

**Decisión:** inspección manual código a código de los 72 eventos de tipo Stop y Warning.

**Justificación:** el filtrado por frecuencia eliminaría fallos críticos de baja ocurrencia —precisamente los más costosos— y retendría ruido frecuente como las pruebas de batería (código 710, 248 ocurrencias). El filtrado por categoría contractual agrupa códigos heterogéneos: la categoría "Warnings" incluye tanto averías reales como alertas de mantenimiento planificado. Solo la auditoría visual con contexto de dominio puede distinguir un fallo técnico real de un evento operativo. No hay algoritmo que lo haga sin ese conocimiento previo.

El resultado fue la eliminación de 20 códigos en tres categorías:

- **Mantenimiento planificado**: pruebas de batería (710, 707), limpieza de aceite hidráulico (5760), contador de horas (5700). Son tareas programadas, no síntomas de degradación.
- **Causas externas**: cortes de red (3570, 3500), hielo (6682, 6690, 6540), sobrefrequencia de red (3585, 3590), viento excesivo (64). No tienen precursores detectables en el SCADA de la turbina.
- **Acciones humanas**: paradas manuales en campo (20) y remoto (21, 25), frenado manual (210), park master (8000). Son comandos deliberados, no fallos.

Dos códigos merecen mención especial por haber sido objeto de decisión explícita:

**Código 675** (`Pitch measuring system 1><2`): inicialmente excluido por considerarse fallo de sensor. Revisado y confirmado que puede ser precursor de fallo mecánico de pitch real. Incluido en la familia `pitch_bat`.

**Código 8400** (`Comm. failure FPM`): excluido definitivamente. Pérdida de comunicación con el módulo de potencia sin correlación con degradación mecánica — aparece también durante mantenimientos de red. Incluir este código contaminaría el etiquetado con falsos positivos sistemáticos.

---

## 3. Familias de fallo descartadas para ML

**Sensores / Anemómetros** (códigos 6515, 6525, 6530, 6620, 6622, 6635): los 86 eventos están concentrados en un único mes —diciembre 2021— lo que indica un incidente puntual de hardware de sensor, no un patrón de degradación aprendible con cinco años de datos. Un modelo entrenado sobre un único incidente no generaliza. La detección se implementa mejor como regla determinista: divergencia entre sensor 1 y sensor 2 superior a 1.5 m/s durante tres intervalos consecutivos → alerta directa, sin ML.

**Torre / Vibración** (códigos 4510, 4520, 4540, 59): dos problemas independientes. Primero, solo 26 eventos en cuatro años es insuficiente estadísticamente para entrenar cualquier modelo. Segundo, y más importante, las oscilaciones de torre son fenómenos que ocurren en segundos y quedan completamente enmascarados en la media de 10 minutos del SCADA. Para detectar este fallo se necesitarían datos de acelerómetro a frecuencia mínima de 1 Hz — fuera del alcance de este dataset.

**Drivetrain** (código 1070): un único evento en todo el período 2018–2022. Sin mínimo estadístico posible para ningún enfoque supervisado.

---

## 4. Lead times por familia

Los lead times son la decisión de diseño con mayor impacto sobre la calidad del etiquetado. El criterio no fue empírico-estadístico sino físico-estadístico combinado: ¿cuántas horas antes del fallo hay señal detectable en los sensores, dado un porcentaje de positivos que permita discriminar?

El porcentaje de positivos en train actúa como restricción práctica: por encima del 30%, el modelo puede alcanzar métricas aceptables prediciendo siempre "pre-fallo" sin aprender ningún patrón real. Por debajo del 2%, hay tan pocas filas positivas que el modelo no tiene suficiente señal para aprender.

| Familia | Lead time final | % positivos en train | Rango evaluado |
|---|:---:|:---:|---|
| `yaw_cable` | 83 h | ~16% | 48h–168h · descartado 168h (47% positivos) |
| `generator` | 127 h | ~17% | 72h–200h |
| `brake_hydro` | 130 h | ~22% | 72h–200h |
| `pitch_bat` | 295 h | ~17% | 168h–400h |

---

## 5. Split temporal estricto vs. split aleatorio

**Alternativa descartada:** `train_test_split` con `shuffle=True` o `StratifiedKFold`.

**Decisión:** split temporal 60/20/20 por posición en el tiempo, sin aleatorización.

**Justificación:** los datos de series temporales tienen dependencia temporal. Una fila de telemetría del lunes predice el estado del miércoles. Si el miércoles cae en train y el lunes en test, el modelo vio en entrenamiento información que cronológicamente era posterior al test — data leakage estructural. La validación cruzada estándar es inaplicable por el mismo motivo. El split temporal garantiza que el modelo solo puede predecir sobre datos que cronológicamente son posteriores a su entrenamiento, exactamente como ocurre en producción.

---

## 6. Exclusión de las 24 horas posteriores a cada fallo

**Decisión:** las 24 horas de telemetría inmediatamente posteriores a cada evento de fallo se eliminan del dataset de entrenamiento.

**Justificación:** en el período de transición post-reparación, los sensores exhiben comportamiento atípico: temperaturas descendiendo desde picos de fallo, presiones recuperándose, sistemas reiniciándose. Incluir esas filas como "estado sano" contaminaría la clase negativa con patrones que son en realidad consecuencia del fallo, no estado normal de operación. El modelo aprendería que "recién después de fallar" es estado sano — exactamente lo contrario de lo que se quiere.

---

## 7. LightGBM vs. otras alternativas

**Alternativas evaluadas:** XGBoost, Random Forest, modelos lineales (Ridge, Lasso).

**Decisión:** LightGBM Regressor.

**Justificación:** en el espacio de features de alta dimensionalidad con muchas rolling statistics correlacionadas, los modelos de boosting por histograma tienen mejor comportamiento que Random Forest (más eficientes en memoria, mejores en features dispersas) y que XGBoost (más rápido en convergencia con `num_leaves` como parámetro principal). Los modelos lineales no capturan las interacciones no lineales entre features de ventana y features de dominio que son centrales en este problema.

El parámetro más crítico por familia fue `num_leaves`: controla la complejidad del árbol y es el principal mecanismo de regularización en datos con desequilibrio de clases. `brake_hydro` usa `num_leaves=31` (árbol más conservador) por tener el mayor desequilibrio relativo.

---

## 8. Features de pendiente descartadas

**Decisión:** las features de pendiente (`slope` sobre ventanas temporales) se calcularon y descartaron tras evaluación.

**Justificación:** AUC univariante de 0.500 en todas las familias —equivalente a predicción aleatoria. La degradación en turbinas eólicas no sigue una rampa monótona que una pendiente pueda capturar: los sensores fluctúan continuamente con el viento, la carga y la temperatura exterior. La señal predictiva está en los valores extremos sostenidos y en la frecuencia de excedencia del baseline histórico, no en la derivada puntual del valor.

---

## 9. Isotonic Regression vs. Platt Scaling para calibración

**Alternativa descartada:** Platt Scaling (regresión logística sobre la salida del regresor).

**Decisión:** Isotonic Regression.

**Justificación:** Platt Scaling asume que la relación entre predicción bruta y valor real es sigmoidal — razonable para clasificadores de margen como SVM, pero sin justificación para la salida de un regresor de árboles. Isotonic Regression aprende cualquier función monótona no paramétrica, lo que la hace más adecuada para corregir la compresión hacia la media que produce LightGBM cuando el número de árboles efectivos es limitado. La única restricción que impone —monotonía— es exactamente la que se desea: una predicción bruta más alta debe corresponder a un valor calibrado más alto.

---

## 10. Métrica principal: Event Recall vs. AUC-ROC o F1

**Alternativas consideradas:** AUC-ROC, F1, Precision@K.

**Decisión:** Event Recall como métrica primaria de selección de modelo.

**Justificación:** AUC-ROC mide la capacidad discriminante fila a fila, ignorando la estructura temporal de los eventos. Un modelo puede tener AUC-ROC de 0.90 y fallar en detectar el 40% de los fallos reales si sus aciertos están concentrados en filas individuales alejadas del evento. F1 tiene el mismo problema: pondera iguales las filas positivas independientemente de si su acierto contribuye a detectar un fallo real o no. Event Recall mide exactamente lo que importa operativamente: de los fallos reales que ocurrieron, cuántos recibieron al menos una alerta antes de tiempo. Un fallo no anticipado tiene un coste concreto —parada no planificada, grúa, reparación de emergencia—. Un fallo anticipado tiene coste cero en comparación. La métrica tiene que reflejar esa asimetría.
