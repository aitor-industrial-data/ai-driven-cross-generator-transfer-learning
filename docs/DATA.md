# Data — AI-Driven Cross-Generator Transfer Learning

Este documento describe el dataset de origen, el esquema de cada capa del pipeline de datos, los Feature Stores por familia y el catálogo completo de fault codes.

---

## Dataset de origen

**Kelmarsh Wind Farm** — parque eólico onshore en Northamptonshire, Reino Unido. Los datos son públicos y provienen de la plataforma Greenbyte, que registra la telemetría SCADA de las turbinas Senvion MM92 del parque.

| Atributo | Valor |
|---|---|
| Turbina de entrenamiento | Kelmarsh Turbine 1 (T1) |
| Turbina de producción | Kelmarsh Turbine 2 (T2) |
| Período de entrenamiento | 2018 – 2022 (5 años) |
| Resolución temporal | 10 minutos |
| Registros totales (T1) | 262.944 filas |
| Columnas SCADA brutas | 303 |
| Eventos de estado en Bronze | 37.230 |

Los archivos CSV originales tienen un formato no estándar: la línea 10 es el header real, precedida por 9 líneas de metadatos prefijadas con `#`. Los nombres de columna contienen caracteres especiales —comas, paréntesis, símbolos `°` y `/`— incompatibles con Spark. El pipeline los convierte a `snake_case` limpio en el paso de ingesta.

Adicionalmente, las columnas de estadísticos intra-intervalo (Max, Min, StdDev por sensor) solo empezaron a registrarse en años posteriores a 2018. Los archivos de distintos años tienen columnas ligeramente distintas por actualizaciones de firmware. La carga usa `unionByName` con `allowMissingColumns=True` para preservar toda la información disponible sin pérdida.

---

## Arquitectura de capas

### Bronze — Datos crudos

Telemetría SCADA tal como viene de Greenbyte, con headers limpios pero sin ninguna otra transformación. Particionada por año en disco. Se trata como inmutable: nunca se sobreescribe.

```
data/bronze/
├── Kelmarsh_SCADA_2018_*/Turbine_Data_Kelmarsh_1_*.csv
├── Kelmarsh_SCADA_2019_*/Turbine_Data_Kelmarsh_1_*.csv
├── Kelmarsh_SCADA_2020_*/Turbine_Data_Kelmarsh_1_*.csv
├── Kelmarsh_SCADA_2021_*/Turbine_Data_Kelmarsh_1_*.csv
└── Kelmarsh_SCADA_2022_*/Turbine_Data_Kelmarsh_1_*.csv
```

En producción, el equivalente para T2 es:
```
s3://ai-driven-cross-generator-transfer-learning/
└── bronze/
    ├── turbine_2_telemetry_clean.parquet/    ← SCADA T2 (ya en Parquet particionado)
    └── turbine_2_status_2026_2030.csv        ← log de eventos de estado T2
```

### Silver — Datos limpios, listos para etiquetar

Resultado del notebook `03_merge_and_cleaning`. Dos artefactos:

**`turbine_1_telemetry_clean.parquet`** — telemetría filtrada a las 46 columnas de sensores utilizables más `timestamp`. 262.944 filas × 47 columnas.

El criterio de selección de columnas fue disponibilidad + causalidad física. El umbral de descarte por nulos fue **> 40%**: por encima de ese nivel, la columna tiene más huecos que datos y no es fiable como feature primaria. Las 3 columnas con nulos entre 10–40% (`blade_angle_*_standard_deviation`, `gear_oil_temperature_max_c`) no se incluyen en Silver pero se usan indirectamente en features de ventana rodante calculadas en el notebook `05_features`.

**`t1_fault_targets_grouped.parquet`** — log de fallos técnicos reales de T1, agrupados por `(timestamp, familia)`. ~330 filas × 6 columnas: `timestamp`, `fault_code`, `message`, `status`, `family`, `count`. La agrupación resuelve el caso de múltiples fallos simultáneos en el mismo intervalo de 10 minutos: se conserva un único evento por familia por timestamp, con el código de mayor criticidad.

### Feature Stores — Features calculadas por familia

Resultado del notebook `05_features`, aplicado sobre el dataset etiquetado. Un Parquet por familia, con todas las features de ventana rodante, features de dominio, `hours_since_last_fault` y columnas target (`hours_to_fault`, `is_pre_fault`).

En producción, los Feature Stores de T2 crecen cada noche con las filas del día. Los de T1 son estáticos —se generaron durante el desarrollo y se subieron a S3 una única vez.

---

## Sensores utilizados por familia

### `yaw_cable`
Sensores brutos: posición de la góndola, dirección del viento, velocidad de viento, corriente de los motores de yaw, contador de vueltas del cable.

Features de dominio calculadas:
- `yaw_error`: desviación absoluta entre posición de góndola y dirección del viento, normalizada a [0°, 180°]
- `yaw_error_wind`: `yaw_error` × velocidad de viento — pondera el desalineamiento por la fuerza que ejerce sobre el cable

### `generator`
Sensores brutos: temperaturas de rodamiento frontal y trasero, temperatura de estátor (×2), temperatura ambiente de góndola, corriente de fase del generador, potencia activa, temperaturas de ventiladores.

Features de dominio calculadas:
- `t_bearing_delta`: temperatura de rodamiento frontal menos temperatura ambiente — elimina el efecto estacional
- `t_rear_bearing_delta`: ídem para rodamiento trasero
- `t_bearing_diff`: diferencia entre rodamiento frontal y trasero — asimetría térmica entre cojinetes
- `t_stator_bearing_diff`: diferencia entre estátor y rodamiento — indica calentamiento diferencial del bobinado

### `brake_hydro`
Sensores brutos: presión hidráulica, temperatura de aceite hidráulico, temperatura de aceite de caja de cambios, estado del acumulador, corriente de freno.

### `pitch_bat`
Sensores brutos: ángulos de pitch de las tres palas (A, B, C), temperatura de gabinete del hub, temperatura ambiente exterior.

Features de dominio calculadas:
- `pitch_asymmetry`: diferencia entre máximo y mínimo de los tres ángulos de pala — asimetría de paso entre palas
- `blade_angle_mean`: media de los tres ángulos — referencia de posición colectiva
- `t_hub_delta`: temperatura de gabinete del hub menos temperatura ambiente exterior — indica pérdida de capacidad de calefacción, síntoma de baterías degradadas en frío

---

## Esquema de rolling features

Para cada sensor se generan **17 features** sobre 4 ventanas temporales (1 h, 6 h, 24 h, 7 días):

| Feature | Descripción |
|---|---|
| `{sensor}_{w}_mean` | Media en ventana — nivel promedio del sensor |
| `{sensor}_{w}_std` | Desviación estándar — dispersión y ruido |
| `{sensor}_{w}_p95` | Percentil 95 — valores extremos sostenidos |
| `{sensor}_{w}_exceedance` | Fracción del tiempo por encima del p90 del baseline — frecuencia de superación del umbral histórico sano |
| `{sensor}_7d_ratio` | Media 7 días / media de los primeros 180 días — desviación relativa respecto al comportamiento sano de referencia |

`min_periods = ventana // 3` permite calcular estadísticos aunque la ventana no esté completamente llena al inicio de la serie. Las filas sin suficiente contexto se imputan con 0 al entrenar.

---

## Catálogo de fault codes

### Familia `yaw_cable`
| Código | Mensaje | Status |
|---|---|---|
| 6052 | High yaw motor current | Warning |
| 6054 | Easy yaw | Warning |
| 6120 | Uncontrolled yaw movement | Stop |
| 6200 | Cable autounwind | Stop |
| 6300 | Yaw error | Stop |

### Familia `generator`
| Código | Mensaje | Status |
|---|---|---|
| 2550 | Overload generator fan 1 | Warning |
| 2650 | Overload generator fan 2 | Warning |
| 2655 | Overload generator fan 3 | Warning |
| 2674 | Overload generator heating | Warning |
| 3000 | Frequency converter not ready | Stop |
| 3110 | Frequency converter error | Stop |
| 3125 | Timeout ready for connection | Stop |
| 3205 | PT100 converter inlet temperature defect | Warning |
| 3220 | Reduced power converter | Warning |

### Familia `brake_hydro`
| Código | Mensaje | Status |
|---|---|---|
| 2000 | Brake pads worn | Warning |
| 2125 | Timeout brake closed | Warning |
| 5510 | Low hydraulic pressure | Stop |
| 5720 | Brake accumulator defect | Warning |

### Familia `pitch_bat`
| Código | Mensaje | Status |
|---|---|---|
| 440 | Repeating error BP 0 | Warning |
| 455 | Repeating error BP52 | Stop |
| 675 | Pitch measuring system 1><2 | Warning |
| 681 | Limit switch error 95° axis 1 | Warning |
| 682 | Limit switch error 95° axis 2 | Warning |
| 683 | Limit switch error 95° axis 3 | Warning |
| 692 | Pitch run-away (hub box v.>=4) | Stop |
| 716 | Battery charge cycle axis 1 error | Warning |
| 717 | Battery charge cycle axis 2 error | Warning |
| 718 | Battery charge cycle axis 3 error | Warning |
| 720 | Pitch batteries charging cycle | Warning |
| 785 | Error brake resistor CHP | Warning |
| 850 | Error lubrication pump pitch | Warning |

### Códigos descartados (ruido operativo)
| Código | Mensaje | Motivo de descarte |
|---|---|---|
| 710 | Battery test | Prueba funcional programada |
| 707 | Stop battery test | Prueba funcional programada |
| 5760 | Hydraulic oil flushing operation | Mantenimiento preventivo |
| 5700 | Max. operation time hydraulic | Contador de horas, no fallo |
| 3590 | Overvoltage | Pico de tensión externo |
| 3570 | Grid error | Causa externa (red) |
| 3500 | Grid loss | Causa externa (red) |
| 6682 | Icing (dev. electr. power) | Efecto secundario de hielo |
| 6690 | Icing (stop) | Causa ambiental |
| 6540 | Icing (anemometer) | Causa ambiental |
| 64 | Max. wind speed | Límite operativo externo |
| 20 | Manual stop - on site | Acción humana deliberada |
| 21 | Manual stop - remote | Acción humana deliberada |
| 25 | Manual stop without login | Acción humana deliberada |
| 210 | Manual brake | Acción humana deliberada |
| 8000 | Park master stop | Comando de control externo |
| 7325 | Time sync. failed (SNTP error) | Error administrativo de red |
| 8400 | Comm. failure FPM | Pérdida de comunicación, sin correlación mecánica |
| 3585 | Maximum grid frequency | Causa externa (red) |
| 68 | Deviation winddirection > 60° | Condición de viento, no degradación |

### Familias descartadas para ML
| Familia | Códigos | Motivo |
|---|---|---|
| Sensores / Anemómetros | 6515, 6525, 6530, 6620, 6622, 6635 | 86 eventos concentrados en un único mes (dic 2021) — incidente puntual no aprendible |
| Torre / Vibración | 4510, 4520, 4540, 59 | Fenómeno de segundos enmascarado en media de 10 min; 26 eventos en 4 años |
| Drivetrain | 1070 | Un único evento en todo el período — sin mínimo estadístico posible |
