
"Simulo el proceso real de una empresa industrial que tiene el historial de fallos de su PLC pero instala una máquina nueva. El modelo entrenado con el historial antiguo sirve de base. Con solo unas semanas de datos de la máquina nueva el modelo se adapta sin empezar de cero. Eso es lo que diferencia la IA del mantenimiento correctivo tradicional."

"Simulo el proceso de una empresa que tiene un parque con generadores industriales síncronos antiguos con historial de fallos documentado y está modernizando con generadores de inducción SCIG de nueva instalación. El modelo entrenado con el historial eléctrico del parque antiguo sirve de base para aprender la firma de fallo del generador nuevo con muchos menos datos."

"Simulo el proceso de una empresa que tiene un parque de maquinaria rotatoria antigua con historial de degradación documentado hasta el fallo. El modelo entrenado con ese historial sirve de base para detectar el fallo prematuro en equipos nuevos de diferente fabricante con muchos menos datos de operación."

El Problema Real: "Cuando una planta gasta millones en cambiar motores antiguos por motores nuevos de alta eficiencia, no puede esperar 3 años a que los motores nuevos fallen para tener datos suficientes y entrenar una IA de mantenimiento predictivo."

La Solución de Datos: "Diseñé un pipeline donde entreno un modelo base (ej. XGBoost o una red neuronal ligera) con el histórico del parque antiguo. Luego, usando Transfer Learning (congelando capas o usando técnicas de adaptación de dominio/Fine-Tuning), re-entreno el modelo con apenas un 5% de datos del parque nuevo. El modelo es capaz de detectar los fallos nuevos desde el primer mes operando."

"El objetivo de este proyecto es resolver un problema crítico en la transición hacia la Industria 4.0: el arranque en frío (Cold Start) del mantenimiento predictivo.

Cuando una planta industrial invierte millones en sustituir maquinaria antigua por equipos modernos y eficientes, se enfrenta a un dilema: los modelos de IA entrenados para predecir fallos ya no sirven porque las firmas de operación han cambiado, y no podemos permitirnos esperar dos años a que las máquinas nuevas rompan para acumular un nuevo historial de fallos.

Para demostrarlo de forma realista, utilicé el dataset Metro-PT, que contiene 7 meses de telemetría real a 1 Hz de un compresor industrial con fallos mecánicos y eléctricos reales sobre el terreno. Como Data Engineer, procesé este flujo masivo de datos aplicando ventanas móviles (Rolling Windows) para capturar la degradación temporal (la tendencia de la vibración y los picos de corriente del motor en el tiempo), evitando así que el modelo tomara decisiones basadas en fotos instantáneas aisladas.

Utilizando el grueso del histórico como 'Parque Antiguo', entrené un modelo base. Después, simulando la llegada del 'Parque Nuevo' con apenas un par de semanas de datos limpiaz, apliqué técnicas de Domain Adaptation / Transfer Learning para transferir el conocimiento del desgaste del motor antiguo al nuevo. El resultado es un pipeline capaz de alertar de anomalías en equipos recién instalados reduciendo la necesidad de datos históricos en más de un 80%."