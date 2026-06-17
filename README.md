<div align="center">

<br>

# AI-Driven Cross-Generator Transfer Learning
### Predictive Maintenance · Cold Start Problem · Wind Turbines

<br>

[![AWS](https://img.shields.io/badge/AWS-Lambda%20%7C%20S3%20%7C%20ECR%20%7C%20ECS%20Fargate-FF9900?style=flat-square&logo=amazonaws&logoColor=white)](https://aws.amazon.com)
[![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![LightGBM](https://img.shields.io/badge/LightGBM-Transfer%20Learning-2ecc71?style=flat-square)](https://lightgbm.readthedocs.io)
[![PySpark](https://img.shields.io/badge/PySpark-ETL%20Pipeline-E25A1C?style=flat-square&logo=apachespark&logoColor=white)](https://spark.apache.org)
[![CI/CD](https://img.shields.io/badge/CI%2FCD-GitHub%20Actions%20%E2%86%92%20ECR%20%E2%86%92%20Lambda-2088FF?style=flat-square&logo=githubactions&logoColor=white)](https://github.com/features/actions)

<br>

<a href="https://ai-driven-cross-generator-transfer-learning.s3.eu-south-2.amazonaws.com/html/dashboard_t2_v2.html">
  <img src="https://img.shields.io/badge/%E2%9A%A1%20VER%20DASHBOARD%20EN%20VIVO%20%E2%86%92-Predicciones%20activas%20%C2%B7%20Turbina%202-22c55e?style=for-the-badge&labelColor=16a34a" alt="Ver Dashboard en vivo" height="45"/>
</a>

<br>
<sub><em>Actualizado cada 24 h · Pipeline serverless en AWS · Datos reales de Kelmarsh Wind Farm</em></sub>

<br><br>

</div>

---

## El problema que resuelve

Un parque eólico renueva una turbina. Los ingenieros tienen años de histórico de la máquina antigua: miles de horas de telemetría SCADA, registros de averías, patrones de desgaste conocidos. Pero el modelo de IA entrenado sobre esos datos **ya no sirve para la máquina nueva**: las firmas operativas han cambiado, los rangos térmicos difieren, los comportamientos antes de fallo son distintos.

La alternativa obvia —esperar a que la nueva turbina acumule su propio historial de averías— puede significar dos años de operación a ciegas. Dos años en los que cualquier fallo no anticipado en el generador, el sistema hidráulico o el pitch supone una parada no planificada de días, grúas de emergencia y pérdidas que en eólico offshore pueden superar los 200.000€ por evento.

Esto es el **Cold Start del mantenimiento predictivo**, y no tiene solución trivial.

---

## La solución: Transfer Learning entre generadores

La hipótesis de este proyecto es que **el conocimiento físico de la degradación es transferible** entre turbinas del mismo tipo, aunque operen en condiciones distintas. Los síntomas previos a un fallo hidráulico —la deriva lenta de presión, los picos de corriente en el freno, la acumulación de calor— siguen la misma física en una turbina y en otra.

El sistema usa la Turbina 1 (T1) con cinco años de histórico como **donante de conocimiento**. La Turbina 2 (T2), recién operativa, arranca protegida desde el primer día.

```
T1 · 5 años de SCADA + averías reales
            │
            │  Entrenamiento inicial
            ▼
     Modelo base T1
            │
            │  Transfer Learning
            │  (T2 arranca sin historial de fallos)
            ▼
  Modelo T1+T2 en producción  ──→  Alertas diarias sobre T2
```

Pero la solución no termina ahí. A medida que T2 opera y acumula sus propios eventos de fallo, el sistema evalúa automáticamente si el modelo beneficia más entrenado con **T1+T2 combinados** o ya solo con **datos propios de T2**. Con el tiempo, T2 habrá acumulado suficiente historial propio para no necesitar a T1. El sistema detecta ese momento y conmuta solo.

---

## Cómo funciona el sistema

**Inferencia diaria** — Cada noche, una Lambda serverless consume el SCADA del día de T2, actualiza los Feature Stores por familia de fallo y publica predicciones de horas-hasta-avería para el día siguiente. El dashboard del operario se actualiza automáticamente.

**Reentrenamiento mensual** — El día 1 de cada mes, un contenedor en ECS Fargate entrena dos versiones del modelo para cada familia:

- Versión A: entrenada únicamente con datos de T2 acumulados hasta la fecha
- Versión B: entrenada con el histórico completo de T1 más los datos de T2

Ambas versiones se evalúan sobre el test set real de T2. La que mejor Event Recall alcanza se despliega automáticamente como modelo en producción. No hay intervención manual.

A medida que T2 acumula meses de operación, se espera que la Versión A vaya ganando terreno progresivamente hasta superar a la Versión B de forma consistente. Ese cruce es la validación empírica central del proyecto: **el Transfer Learning es útil exactamente mientras los datos propios son insuficientes**, y el sistema lo detecta y gestiona solo.

La fiabilidad real del sistema no se mide en el test set inicial —que tiene pocos fallos de T2 y poco tiempo de operación—, sino en la calidad de sus predicciones sobre fallos reales a medida que el parque opera. Un modelo de mantenimiento predictivo sobre equipos industriales se afina durante años, no durante semanas. Lo que este proyecto garantiza es que T2 **no arranca desde cero**: hereda una base de conocimiento sólida que de otro modo requeriría años de averías propias para construir.

---

## Por qué este problema es difícil

Los fallos industriales en turbinas eólicas no son eventos simples de clasificar. Son raros —una familia de fallo puede tener una docena de eventos en cinco años—, heterogéneos —el mismo código puede tener causas distintas en distintas épocas del año— y con un lag temporal no trivial: un fallo de pitch puede gestarse 12 días antes de manifestarse; uno hidráulico, menos de una semana.

El SCADA no etiqueta fallos, etiqueta *paradas*. Distinguir una parada por mantenimiento programado de una avería real requiere una auditoría código a código sobre más de 75 tipos de evento distintos. Los sensores tienen huecos, derivas y headers malformados. Y los regressores sobre series temporales industriales comprimen sus predicciones hacia la media si no se calibran explícitamente, produciendo alertas que llegan siempre tarde o siempre pronto.

Cada una de esas fricciones está resuelta en este pipeline. Pero resolver la ingeniería no elimina la limitación más fundamental: **los fallos industriales son eventos raros**. T1 acumuló cinco años de operación para tener suficientes eventos de cada familia. T2 lleva meses. El sistema está diseñado para ser útil desde el primer día, pero su fiabilidad real crecerá con cada fallo nuevo que registre, con cada reentrenamiento mensual, con cada año adicional de operación. La arquitectura de mejora continua no es un complemento — es parte central del diseño.

---

## Arquitectura AWS

```
GitHub push
    │
    └─→  GitHub Actions (OIDC)
              │
              ├─→  ECR  ←────────────────────────────────────────┐
              │     └─ imagen inferencia (Lambda / python:3.12)  │
              │     └─ imagen reentrenamiento (Fargate / slim)   │
              │                                                  │
              └─→  Lambda t2-inference  ←── EventBridge (diario) │
                        │                                        │
              ECS Fargate t2-retrain  ←── EventBridge (día 1)────┘
                        │
                        ▼
              S3: bronze/ · models/ · html/
                        │
                        └─→  Dashboard HTML (acceso público)
```

Región `eu-south-2 (Madrid)` · Imágenes desplegadas por digest · OIDC sin credenciales de larga duración

---

## Pipeline de desarrollo (T1 · Kelmarsh 2018–2022)

| Notebook | Qué resuelve |
|---|---|
| `01_eda_status_and_events` | Auditoría de 75+ códigos de evento → 52 válidos. Descarte documentado de 3 familias |
| `02_eda_telemetry_and_sensors` | ~300 señales SCADA · limpieza de headers `#`-prefijados · selección por disponibilidad física |
| `03_merge_and_cleaning` | Bronze → Silver · exclusión de 24 h post-fallo · gestión de columnas variables por año |
| `04_labeling` | Etiquetado temporal `hours_to_fault` con lead time por familia · split 60/20/20 estrictamente temporal |
| `05_features` | Features de dominio (yaw error ponderado por viento, deltas térmicos) + rolling statistics 7 días |
| `06_train` ×4 | LightGBM regresor independiente por familia · evaluación por Event Recall sobre eventos reales |
| `07_calibration` | Isotonic Regression sobre salida del regresor · convierte predicciones en probabilidades accionables |

---

## Documentación técnica

- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — Infraestructura AWS completa y flujo de datos
- [`docs/ML_DESIGN.md`](docs/ML_DESIGN.md) — Diseño del pipeline ML: familias, etiquetado, features, métricas
- [`docs/PRODUCTION.md`](docs/PRODUCTION.md) — Operación: inferencia diaria, reentrenamiento, cold start
- [`docs/DECISIONS.md`](docs/DECISIONS.md) — Registro de decisiones técnicas y alternativas descartadas
- [`docs/DATA.md`](docs/DATA.md) — Dataset, esquema Bronze/Silver, Feature Stores, catálogo de fault codes