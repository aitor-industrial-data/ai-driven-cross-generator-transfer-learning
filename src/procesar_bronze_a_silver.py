import os
import glob
import logging
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

# Configuración del Logger profesional bajo estándar industrial
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("Kelmarsh-DataEngine")

# 1. Inicialización de la sesión de Spark optimizada (10 hilos, 16GB RAM)
spark = SparkSession.builder \
    .appName("Kelmarsh-WindFarm-PdM-Silver") \
    .master("local[10]") \
    .config("spark.driver.memory", "16g") \
    .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
    .getOrCreate()

# Reducir el ruido de logs internos de Spark
spark.sparkContext.setLogLevel("WARN")

# 2. Definición de rutas absolutas (Arquitectura Medallón)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BRONZE_DIR = os.path.join(BASE_DIR, "data", "bronze")
SILVER_DIR = os.path.join(BASE_DIR, "data", "silver")

HISTORICAL_TRAINING_DIR = os.path.join(SILVER_DIR, "historical_baseline_training")
STREAMING_SIMULATION_DIR = os.path.join(SILVER_DIR, "target_streaming_simulation")

os.makedirs(HISTORICAL_TRAINING_DIR, exist_ok=True)
os.makedirs(STREAMING_SIMULATION_DIR, exist_ok=True)

# ==========================================
# FASE 1: LOGS DE ESTADO (STATUS)
# ==========================================
logger.info("🔍 Fase 1: Localizando y parseando logs de estado (Status) para la Turbina 01...")

status_files = glob.glob(f"{BRONZE_DIR}/**/Status_Kelmarsh_1_*.csv", recursive=True)
if not status_files:
    status_files = glob.glob(f"{BRONZE_DIR}/Status_Kelmarsh_1_*.csv")

if not status_files:
    logger.error(f"No se encontraron archivos de Status para la Turbina 1 en {BRONZE_DIR}")
    raise FileNotFoundError(f"❌ No status log files found for Turbine 1 in {BRONZE_DIR}")

status_df = spark.read \
    .option("header", "True") \
    .option("inferSchema", "True") \
    .option("comment", "#") \
    .csv(status_files)

# Extracción de timestamps de fallos usando la columna real 'Timestamp start'
failure_timestamps_df = status_df.filter(
    (~F.lower(F.col("Status")).contains("normal")) & 
    (~F.lower(F.col("Status")).contains("ok"))
).select(F.to_timestamp("Timestamp start").alias("failure_timestamp")).distinct()

# Colectar marcas de tiempo para la lógica de ventana
failure_list = [row["failure_timestamp"] for row in failure_timestamps_df.collect() if row["failure_timestamp"] is not None]
logger.info(f"⚠️ Detectados {len(failure_list)} eventos de anomalía/fallo en el histórico.")

# ==========================================
# FASE 2: HISTÓRICO SCADA (TURBINA 1)
# ==========================================
logger.info("🔄 Fase 2: Procesando Dataset 1 - Historical Baseline Training (Turbina 1: Años de desgaste)...")

todos_los_scada_t01 = glob.glob(f"{BRONZE_DIR}/**/Turbine_Data_Kelmarsh_1_*.csv", recursive=True)
if not todos_los_scada_t01:
    todos_los_scada_t01 = glob.glob(f"{BRONZE_DIR}/Turbine_Data_Kelmarsh_1_*.csv")

# Filtro flexible: Excluimos cualquier archivo que haga referencia al periodo de simulación de 2016
scada_historical_files = [f for f in todos_los_scada_t01 if "2016-01-03" not in f and "_2016" not in f]
scada_historical_files = list(set(scada_historical_files))

if not scada_historical_files:
    logger.error("No se encontraron archivos SCADA históricos para la Turbina 1.")
    raise FileNotFoundError("❌ No historical SCADA files found for Turbine 1.")

logger.info(f"📂 Archivos históricos encontrados para procesar: {len(scada_historical_files)}")
for f in sorted(scada_historical_files):
    logger.info(f"   -> Incluyendo en histórico: {os.path.basename(f)}")

historical_scada_df = spark.read \
    .option("header", "True") \
    .option("inferSchema", "True") \
    .option("comment", "#") \
    .csv(scada_historical_files)

primer_columna_hist = historical_scada_df.columns[0]
historical_scada_df = historical_scada_df.withColumnRenamed(primer_columna_hist, "Timestamp")
historical_scada_df = historical_scada_df.withColumn("Timestamp", F.to_timestamp("Timestamp"))

logger.info("⏳ Aplicando ventana temporal de 24 horas mediante Broadcast Join optimizado...")
fallos_df = spark.createDataFrame([(f,) for f in failure_list], ["fecha_fallo"])
fallos_df = fallos_df.withColumn("fecha_fallo", F.to_timestamp("fecha_fallo")) \
                     .withColumn("inicio_degradacion", F.col("fecha_fallo") - F.expr("INTERVAL 24 HOURS"))

historical_scada_df = historical_scada_df.join(
    F.broadcast(fallos_df),
    (F.col("Timestamp") >= F.col("inicio_degradacion")) & (F.col("Timestamp") <= F.col("fecha_fallo")),
    "left"
)

historical_scada_df = historical_scada_df.withColumn(
    "target", 
    F.when(F.col("fecha_fallo").isNotNull(), 1).otherwise(0)
).drop("fecha_fallo", "inicio_degradacion")

historical_output_path = os.path.join(HISTORICAL_TRAINING_DIR, "turbine_01_historical_5y.parquet")
historical_scada_df.write.mode("overwrite").parquet(historical_output_path)
logger.info(f"💾 Dataset 1 guardado correctamente en: {historical_output_path}")


# ==========================================
# FASE 3: SIMULACIÓN STREAMING (TURBINA 2)
# ==========================================
logger.info("🆕 Fase 3: Procesando Dataset 2 - Target Stream Simulation (Turbine 2: Operación limpia 2016)...")

# Patrón omnidireccional: Buscamos cualquier SCADA de la Turbina 2 que contenga "2016" en la ruta o nombre
todos_los_scada_t02 = glob.glob(f"{BRONZE_DIR}/**/Turbine_Data_Kelmarsh_2_*.csv", recursive=True)
if not todos_los_scada_t02:
    todos_los_scada_t02 = glob.glob(f"{BRONZE_DIR}/Turbine_Data_Kelmarsh_2_*.csv")

# Nos quedamos específicamente con el archivo que cubre el año 2016
scada_clean_files = [f for f in todos_los_scada_t02 if "2016" in f]

if not scada_clean_files:
    logger.error("No se encontró el archivo SCADA de simulación para la Turbina 2 (2016).")
    raise FileNotFoundError("❌ Target streaming simulation file for Turbine 2 (2016) not found.")

logger.info(f"📂 Archivo de simulación streaming detectado: {os.path.basename(scada_clean_files[0])}")

clean_scada_df = spark.read \
    .option("header", "True") \
    .option("inferSchema", "True") \
    .option("comment", "#") \
    .csv(scada_clean_files)

primer_columna_clean = clean_scada_df.columns[0]
clean_scada_df = clean_scada_df.withColumnRenamed(primer_columna_clean, "Timestamp")

clean_scada_df = clean_scada_df.withColumn("Timestamp", F.to_timestamp("Timestamp")) \
                               .withColumn("target", F.lit(0))

streaming_output_path = os.path.join(STREAMING_SIMULATION_DIR, "turbine_02_clean_1y.parquet")
clean_scada_df.write.mode("overwrite").parquet(streaming_output_path)
logger.info(f"💾 Dataset 2 guardado correctamente en: {streaming_output_path}")

logger.info("🏁 Pipeline Medallón de Bronze a Silver completado con éxito y blindado contra anomalías de texto.")
spark.stop()