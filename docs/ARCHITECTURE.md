# Architecture — AI-Driven Cross-Generator Transfer Learning

Este documento describe la infraestructura del sistema en producción: qué componente hace qué, cómo se comunican, y por qué se eligió cada servicio.

---

## Visión general

El sistema es completamente serverless y orientado a eventos. No hay servidores que mantener, no hay procesos permanentes en ejecución. Cada componente se activa por un trigger, ejecuta su trabajo, escribe en S3 y termina. El estado global del sistema vive íntegramente en S3.

```
┌──────────────────────────────────────────────────────────────────┐
│  DESARROLLO                                                      │
│                                                                  │
│  Notebooks (local)  ──→  Modelos T1 + Feature Stores T1          │
│                                    │                             │
│                                    ▼                             │
│                          S3: models/t1_*                         │
└──────────────────────────────────────────────────────────────────┘
                                     │
                                     │ (artefactos estáticos, subidos una vez)
                                     ▼
┌──────────────────────────────────────────────────────────────────┐
│  PRODUCCIÓN                                                      │
│                                                                  │
│  GitHub push ──→ GitHub Actions (OIDC)                           │
│                       ├──→ ECR: imagen inferencia                │
│                       └──→ ECR: imagen reentrenamiento           │
│                                    │                             │
│              EventBridge (diario)  │  EventBridge (día 1/mes)    │
│                       │            │            │                │
│                       ▼            │            ▼                │
│              Lambda t2-inference   │   ECS Fargate t2-retrain    │
│                       │            │            │                │
│                       └────────────┴────────────┘                │
│                                    │                             │
│                                    ▼                             │
│                    S3: bronze/ · models/ · html/                 │
│                                    │                             │
│                                    ▼                             │
│                       Dashboard HTML (público)                   │
└──────────────────────────────────────────────────────────────────┘
```

Región: `eu-south-2 (Madrid)`  
Bucket: `ai-driven-cross-generator-transfer-learning`

---

## Componentes

### S3 — Estado global del sistema

S3 es la única fuente de verdad. Todos los componentes leen de S3 al inicio de su ejecución y escriben en S3 al terminar. Esto garantiza que cualquier componente puede fallar y reintentar sin pérdida de estado, y que los componentes son completamente independientes entre sí.

```
s3://ai-driven-cross-generator-transfer-learning/
│
├── bronze/
│   ├── turbine_2_telemetry_clean.parquet/    ← SCADA T2 particionado
│   └── turbine_2_status_2026_2030.csv        ← eventos de estado T2
│
├── models/
│   ├── t1_features_{family}.parquet          ← Feature Stores T1 (estáticos)
│   ├── t2_features_{family}.parquet          ← Feature Stores T2 (crecen cada noche)
│   ├── t1_model_{family}.pkl                 ← modelos base T1 (fallback)
│   ├── t2_model_{family}.pkl                 ← modelos en producción
│   ├── turbine_2_baseline.json               ← media y p90 por sensor (180 días)
│   ├── turbine_2_fault_log.csv               ← fallos técnicos reales de T2
│   ├── t2_predictions_log.csv                ← histórico de predicciones diarias
│   ├── t2_retrain_results.json               ← último reentrenamiento (dashboard)
│   └── t2_retrain_log.csv                    ← histórico acumulativo de reentrenamientos
│
└── html/
    └── dashboard_t2_v2.html                  ← dashboard del operario (acceso público)
```

---

### Lambda — Inferencia diaria

**Imagen base:** `public.ecr.aws/lambda/python:3.12`  
**Dependencia nativa:** `libgomp` (requerida por LightGBM), instalada vía `microdnf`  
**Handler:** `t2_daily_inference_serverless.handler`  
**Timeout:** 300 s  
**Trigger:** EventBridge, cron diario

La imagen Lambda usa la imagen oficial de AWS para Python 3.12 con el runtime Lambda incluido. Esto garantiza compatibilidad con el entorno de ejecución de Lambda sin capas adicionales. `libgomp` es la única dependencia nativa: LightGBM la requiere para paralelismo OpenMP y no está incluida en la imagen base.

La imagen se construye para `linux/amd64` con `--provenance=false` para evitar manifiestos multi-plataforma que ECR rechaza en este contexto. Se despliega por digest —no por tag `:latest`— para que cada actualización sea trazable e irreversible.

---

### ECS Fargate — Reentrenamiento mensual

**Imagen base:** `public.ecr.aws/docker/library/python:3.12-slim`  
**Dependencia nativa:** `libgomp1` vía `apt-get`  
**Cluster:** `sunsaver-cluster`  
**Task definition:** `t2-retrain:latest`  
**Trigger:** EventBridge, día 1 de cada mes  
**Red:** `awsvpcConfiguration` con `assignPublicIp=ENABLED` para acceso a S3 y ECR sin NAT Gateway

El reentrenamiento no puede ejecutarse en Lambda: el entrenamiento LightGBM sobre los Feature Stores combinados de T1+T2 excede el límite de 15 minutos de Lambda y requiere más memoria de la disponible en el tier estándar. Fargate resuelve ambas restricciones —tiempo y memoria ilimitados— con el mismo modelo de facturación por uso: el contenedor se levanta, entrena, escribe en S3 y termina. No hay coste en reposo.

La imagen usa `python:3.12-slim` en lugar de la imagen Lambda porque Fargate no necesita el runtime Lambda. `slim` reduce el tamaño de imagen eliminando herramientas de sistema innecesarias.

---

### ECR — Registro de imágenes

Ambas imágenes viven en repositorios ECR independientes en `eu-south-2`. El despliegue siempre referencia el digest SHA256 de la imagen, no el tag. Esto garantiza que una ejecución de Lambda o Fargate use exactamente la imagen que pasó por CI, no una versión posterior que pudiera haber sido sobreescrita en el tag.

---

### EventBridge — Triggers temporales

Dos reglas de schedule independientes:

| Regla | Expresión cron | Target |
|---|---|---|
| Inferencia diaria | `cron(0 21 * * ? *)` | Lambda `t2-inference-lambda` |
| Reentrenamiento mensual | `cron(0 6 1 * ? *)` | ECS Fargate `t2-retrain` |

La inferencia diaria ejecuta a las 21:00 UTC para garantizar que el SCADA del día esté disponible en Bronze. El reentrenamiento ejecuta a las 06:00 UTC del día 1 para aprovechar el valle de carga de AWS y tener los modelos actualizados antes del turno de mañana.

---

### GitHub Actions — CI/CD con OIDC

El pipeline de despliegue no usa credenciales AWS de larga duración. Usa OIDC: GitHub actúa como proveedor de identidad y asume el rol IAM `github-t2-lambda-role` con una sesión temporal de corta duración generada en el momento del workflow.

El workflow se activa únicamente en push a `main` con cambios en rutas relevantes (`src/`, `requirements.txt`, `Dockerfile*`). Los pasos son:

```
1. Checkout del repositorio
2. Configurar credenciales AWS via OIDC  (sin secrets almacenados)
3. Login en ECR
4. docker buildx build --platform linux/amd64 --provenance=false --push
5. aws lambda update-function-code --image-uri <digest>
```

El flag `--provenance=false` elimina el manifiesto de attestation que ECR rechaza al actualizar Lambda por imagen. El despliegue por digest garantiza que Lambda ejecute exactamente la imagen construida en ese workflow, no la última por tag.

---

## Flujo de datos completo

```
SCADA T2 (simulado hasta 2030, inyectado día a día en Bronze)
    │
    ▼
Lambda · Paso 1: lee telemetría Bronze hasta timestamp=ahora
    │
    ▼
Lambda · Pasos 2-3: carga baseline T2 y fault log T2 desde S3
    │
    ▼
Lambda · Pasos 4-5: calcula features nuevas, append a Feature Stores T2 en S3
    │
    ▼
Lambda · Paso 6: carga modelo (t2 si existe, t1 como fallback)
         aplica LGBMRegressor → IsotonicRegression → predicción calibrada
    │
    ▼
Lambda · Paso 7: append a predictions_log.csv en S3
    │
    ▼
Dashboard HTML: lee predictions_log.csv desde S3 al abrirse
    │
    ▼
Operario: ve estado actual y tendencia de las 4 familias


── día 1 de cada mes ─────────────────────────────────────────────

Fargate · Paso 0: actualiza fault_log T2 desde eventos SCADA Bronze
    │
    ▼
Fargate · Paso 1: carga Feature Stores T1 desde S3
    │
    ▼
Fargate · Paso 2: etiqueta Feature Stores T2, entrena Versión A (T2 solo)
                  y Versión B (T1+T2), evalúa ambas sobre test set T2
    │
    ▼
Fargate · Paso 3: despliega ganador como t2_model_{family}.pkl en S3
                  guarda resultados en retrain_results.json y retrain_log.csv
```

---

## Decisiones de infraestructura

**Por qué S3 como única fuente de estado y no una base de datos:** los volúmenes de datos son manejables en Parquet/CSV. Una base de datos añade coste fijo mensual, gestión de VPC, backups y conexiones. S3 es serverless, virtualmente infinito, y con consistencia eventual suficiente para este patrón de escritura-única-por-ejecución.

**Por qué no Step Functions para orquestar los pasos:** cada ejecución de Lambda es lineal y autocontenida. Step Functions añade complejidad de definición de estado machine, coste por transición y superficie de error adicional sin beneficio real en este caso de uso.

**Por qué OIDC y no IAM keys en secrets de GitHub:** las credenciales de larga duración en secrets de repositorio son un riesgo de seguridad si el repositorio se expone o un colaborador tiene acceso no autorizado. OIDC emite tokens temporales de minutos de duración, scoped al workflow y al repositorio específico.
