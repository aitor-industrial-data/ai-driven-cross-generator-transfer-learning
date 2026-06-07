"""
Script de limpieza autoejecutable para datos SCADA de Kelmarsh.
Convierte archivos CSV crudos (bronze) a Parquet limpio (silver).
"""

import os
import sys
import glob
import re
import logging

# ---------------------------------------------------------------------------
# CONFIGURACIÓN DE LOGGING (Solo por pantalla)
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

logger.info("=" * 70)
logger.info("INICIO DEL SCRIPT DE LIMPIEZA - Kelmarsh SCADA")
logger.info("=" * 70)

# ---------------------------------------------------------------------------
# IMPORTS DE PYSPARK (con try/except robusto)
# ---------------------------------------------------------------------------
try:
    from pyspark.sql import SparkSession
    from pyspark.sql import functions as F
    logger.info("✅ PySpark importado correctamente.")
except ImportError as e:
    logger.error("❌ No se pudo importar PySpark: %s", e)
    logger.error("   Asegúrate de tener PySpark instalado: pip install pyspark")
    sys.exit(1)

# ---------------------------------------------------------------------------
# CONSTANTES Y CONFIGURACIÓN
# ---------------------------------------------------------------------------
from pathlib import Path

BASE_DIR = str(Path(__file__).resolve().parent.parent)
BRONZE_DIR = os.path.join(BASE_DIR, "data", "bronze")
SILVER_DIR = os.path.join(BASE_DIR, "data", "silver")
TEMP_DIR = os.path.join(BASE_DIR, "data", "temp_spark")

NUMBER_TURBINE = 2
TARGET_YEARS = ["2018", "2019", "2020", "2021", "2022"]

os.makedirs(SILVER_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)

logger.info("📁 Directorio base:    %s", BASE_DIR)
logger.info("📁 Directorio bronze:  %s", BRONZE_DIR)
logger.info("📁 Directorio silver:  %s", SILVER_DIR)
logger.info("📁 Directorio temp:    %s", TEMP_DIR)
logger.info("📅 Años objetivo:      %s", ", ".join(TARGET_YEARS))

# ---------------------------------------------------------------------------
# 1. INICIALIZAR SPARK SESSION
# ---------------------------------------------------------------------------
logger.info("")
logger.info("-" * 70)
logger.info("PASO 1/8: Inicializando Spark Session")
logger.info("-" * 70)

try:
    spark = SparkSession.builder \
        .appName("Kelmarsh-Cleaning-Script") \
        .master("local[6]") \
        .config("spark.driver.memory", "8g") \
        .config("spark.sql.adaptive.enabled", "true") \
        .getOrCreate()
    logger.info("✅ Spark Session creada correctamente.")
except Exception as e:
    logger.error("❌ Error al crear Spark Session: %s", e)
    sys.exit(1)

# ---------------------------------------------------------------------------
# 2. FUNCIÓN DE LIMPIEZA DE HEADER
# ---------------------------------------------------------------------------
def clean_header_line(line):
    """
    Limpia una línea de header de CSV:
    - Reemplaza comas dentro de comillas por guiones.
    - Quita prefijo '# ' o '#'.
    - Normaliza nombres de columnas: minúsculas, underscores, sin caracteres especiales.
    - Renombra 'date_and_time' -> 'timestamp'.
    """
    result = []
    in_quotes = False

    for char in line:
        if char == '"':
            in_quotes = not in_quotes
        elif char == ',' and in_quotes:
            result.append('-')
        else:
            result.append(char)

    cleaned = ''.join(result)
    if cleaned.startswith('# '):
        cleaned = cleaned[2:]
    elif cleaned.startswith('#'):
        cleaned = cleaned[1:]

    columns = cleaned.split(',')
    clean_columns = []
    for col in columns:
        col_clean = re.sub(r'[^a-zA-Z0-9_ ]', '', col)
        col_clean = col_clean.replace(' ', '_')
        col_clean = re.sub(r'_+', '_', col_clean)
        col_clean = col_clean.strip('_')
        col_clean = col_clean.lower()

        if col_clean == 'date_and_time':
            col_clean = 'timestamp'

        clean_columns.append(col_clean)

    return ','.join(clean_columns)

# ---------------------------------------------------------------------------
# 3. BUSCAR Y ANALIZAR ARCHIVOS
# ---------------------------------------------------------------------------
logger.info("")
logger.info("-" * 70)
logger.info("PASO 2/8: Buscando archivos CSV en bronze/")
logger.info("-" * 70)

try:
    all_files = []
    for year in TARGET_YEARS:
        pattern = os.path.join(
            BRONZE_DIR,
            f"Kelmarsh_SCADA_{year}_*",
            f"Turbine_Data_Kelmarsh_{NUMBER_TURBINE}_*.csv"
        )
        matched = sorted(glob.glob(pattern))
        all_files.extend(matched)
        logger.info("   %s -> %d archivos encontrados", year, len(matched))

    total_files = len(all_files)
    logger.info("📁 Total de archivos encontrados: %d", total_files)

    if total_files == 0:
        logger.error("❌ No se encontraron archivos CSV. Verifica la ruta: %s", BRONZE_DIR)
        spark.stop()
        sys.exit(1)

except Exception as e:
    logger.error("❌ Error al buscar archivos: %s", e)
    spark.stop()
    sys.exit(1)

# ---------------------------------------------------------------------------
# 4. EXTRAER HEADERS LIMPIOS
# ---------------------------------------------------------------------------
logger.info("")
logger.info("-" * 70)
logger.info("PASO 3/8: Extrayendo y limpiando headers")
logger.info("-" * 70)

files_info = []

try:
    for file_path in all_files:
        filename = os.path.basename(file_path)

        with open(file_path, 'r', encoding='utf-8') as f:
            for line_number, line in enumerate(f, 1):
                if line_number == 10 and 'Date and time' in line:
                    cleaned_header = clean_header_line(line)
                    files_info.append({
                        'path': file_path,
                        'filename': filename,
                        'header': cleaned_header,
                        'skip_lines': 9
                    })
                    logger.info("   ✅ %s | Header: %s...", filename, cleaned_header[:60])
                    break
                elif line_number > 10:
                    logger.warning("   ⚠️  %s | No se encontró header en línea 10", filename)
                    break

    logger.info("📄 Total de archivos con header válido: %d", len(files_info))

    if len(files_info) == 0:
        logger.error("❌ Ningún archivo tiene un header válido en la línea 10.")
        spark.stop()
        sys.exit(1)

except Exception as e:
    logger.error("❌ Error al procesar headers: %s", e)
    spark.stop()
    sys.exit(1)

# ---------------------------------------------------------------------------
# 5. CARGAR CADA ARCHIVO CON SPARK (limpiar comentarios, poner header)
# ---------------------------------------------------------------------------
logger.info("")
logger.info("-" * 70)
logger.info("PASO 4/8: Cargando archivos con Spark")
logger.info("-" * 70)

spark_dfs = []

try:
    for info in files_info:
        file_path = info['path']
        header = info['header']
        filename = info['filename']

        logger.info("🔹 Procesando: %s", filename)

        # Leer archivo original, saltar las 10 primeras líneas (comentarios + header original)
        # y poner el header limpio como primera línea
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()[10:]  # saltar las 10 primeras

        # Insertar header limpio
        lines.insert(0, header + '\n')

        # Escribir archivo temporal limpio
        temp_file = os.path.join(TEMP_DIR, f"temp_{filename}")
        with open(temp_file, 'w', encoding='utf-8') as f:
            f.writelines(lines)

        # Leer con Spark CSV
        sdf = spark.read \
            .option("header", "true") \
            .option("inferSchema", "true") \
            .csv(temp_file)

        row_count = sdf.count()
        col_count = len(sdf.columns)
        spark_dfs.append(sdf)

        logger.info("   ✅ %d filas x %d columnas", row_count, col_count)

    logger.info("📊 Total de DataFrames cargados: %d", len(spark_dfs))

except Exception as e:
    logger.error("❌ Error al cargar archivos con Spark: %s", e)
    spark.stop()
    sys.exit(1)

# ---------------------------------------------------------------------------
# 6. UNIR TODOS LOS DATAFRAMES
# ---------------------------------------------------------------------------
logger.info("")
logger.info("-" * 70)
logger.info("PASO 5/8: Uniendo todos los DataFrames")
logger.info("-" * 70)

try:
    if len(spark_dfs) == 1:
        turbine_data_df = spark_dfs[0]
        logger.info("   ℹ️  Solo un archivo, no es necesario unir.")
    else:
        turbine_data_df = spark_dfs[0]
        for i, sdf in enumerate(spark_dfs[1:], start=2):
            turbine_data_df = turbine_data_df.unionByName(sdf, allowMissingColumns=True)
            logger.info("   ➕ Unido DataFrame %d/%d", i, len(spark_dfs))

    total_records = turbine_data_df.count()
    total_columns = len(turbine_data_df.columns)
    logger.info("📊 DataFrame unificado: %d registros x %d columnas", total_records, total_columns)

except Exception as e:
    logger.error("❌ Error al unir DataFrames: %s", e)
    spark.stop()
    sys.exit(1)

# ---------------------------------------------------------------------------
# 7. SELECCIONAR COLUMNAS BASE Y CALCULAR NULOS
# ---------------------------------------------------------------------------
logger.info("")
logger.info("-" * 70)
logger.info("PASO 6/8: Filtrando columnas base y calculando nulos")
logger.info("-" * 70)

# Columnas base para TODOS los modelos — <10% nulos, disponibles siempre
COLS_BASE = [
    "timestamp",
    # Viento
    "wind_speed_ms", "wind_speed_standard_deviation_ms",
    "wind_speed_sensor_1_ms", "wind_speed_sensor_2_ms",
    "wind_direction", "wind_direction_standard_deviation",
    "nacelle_position", "nacelle_position_standard_deviation",
    "vane_position_12",
    # Potencia
    "power_kw", "power_standard_deviation_kw",
    "power_factor_cosphi", "reactive_power_kvar",
    "grid_voltage_v", "grid_frequency_hz",
    # Generador y tren
    "generator_rpm_rpm", "generator_rpm_standard_deviation_rpm",
    "rotor_speed_rpm",
    "drive_train_acceleration_mmss",
    # Temperaturas
    "generator_bearing_front_temperature_c", "generator_bearing_rear_temperature_c",
    "generator_bearing_front_temperature_max_c", "generator_bearing_rear_temperature_max_c",
    "nacelle_temperature_c", "nacelle_temperature_max_c",
    "nacelle_ambient_temperature_c",
    "ambient_temperature_converter_c",
    "front_bearing_temperature_c", "rear_bearing_temperature_c",
    "gear_oil_temperature_c",
    "gear_oil_inlet_temperature_c",
    "stator_temperature_1_c",
    "temp_top_box_c",
    # Hidráulico
    "gear_oil_inlet_pressure_bar", "gear_oil_pump_pressure_bar",
    # Cable y pitch
    "cable_windings_from_calibration_point",
    "blade_angle_pitch_position_a", "blade_angle_pitch_position_b", "blade_angle_pitch_position_c",
    "motor_current_axis_1_a", "motor_current_axis_2_a", "motor_current_axis_3_a",
    "temperature_motor_axis_1_c", "temperature_motor_axis_2_c", "temperature_motor_axis_3_c",
    # Partículas metálicas
    "metal_particle_count",
]

try:
    # Filtrar columnas que existen
    cols_presentes = [c for c in COLS_BASE if c in turbine_data_df.columns]
    missing_cols = [c for c in COLS_BASE if c not in turbine_data_df.columns]

    if missing_cols:
        logger.warning("⚠️  Columnas no encontradas (se omiten): %s", ", ".join(missing_cols))

    turbine_data_df = turbine_data_df.select(*cols_presentes)
    logger.info("📋 Columnas seleccionadas: %d de %d solicitadas", len(cols_presentes), len(COLS_BASE))

    # Calcular % de nulos + NaN por columna (excluyendo timestamp)
    total_filas = turbine_data_df.count()
    cols_para_nulos = [c for c in cols_presentes if c != "timestamp"]

    if cols_para_nulos:
        nulos_df = turbine_data_df.select([
            (
                F.count(F.when(F.col(c).isNull() | F.isnan(F.col(c)), c)) / total_filas * 100
            ).alias(c)
            for c in cols_para_nulos
        ])

        nulos_row = nulos_df.collect()[0].asDict()
        nulos_ordenados = sorted(nulos_row.items(), key=lambda x: x[1], reverse=True)

        logger.info("")
        logger.info("📊 Top 10 columnas con más nulos + NaN:")
        for col_name, pct in nulos_ordenados[:10]:
            logger.info("   %-50s : %6.2f%%", col_name, pct)
    else:
        logger.info("   ℹ️  No hay columnas para calcular nulos (solo timestamp).")

    logger.info("")
    logger.info("🎯 Total de registros procesados: %d", total_filas)

except Exception as e:
    logger.error("❌ Error al filtrar columnas o calcular nulos: %s", e)
    spark.stop()
    sys.exit(1)

# ---------------------------------------------------------------------------
# 8. DESPLAZAMIENTO DE FECHAS (SUMAR 8 AÑOS)
# ---------------------------------------------------------------------------
logger.info("")
logger.info("-" * 70)
logger.info("PASO 7/8: Desplazando marcas de tiempo (+8 años)")
logger.info("-" * 70)

try:
    # Asegurar que timestamp sea tratado como TimestampType en PySpark
    turbine_data_df = turbine_data_df.withColumn("timestamp", F.to_timestamp("timestamp"))
    
    # Obtener rangos antes del cambio para el log
    rango_original = turbine_data_df.select(F.min("timestamp"), F.max("timestamp")).collect()[0]
    logger.info("📅 Rango original del SCADA: %s  →  %s", rango_original[0], rango_original[1])

    # Aplicar el incremento de 8 años usando expresiones SQL de Spark
    turbine_data_df = turbine_data_df.withColumn("timestamp", F.col("timestamp") + F.expr("INTERVAL 8 YEARS"))
    
    # Obtener rangos modificados
    rango_desplazado = turbine_data_df.select(F.min("timestamp"), F.max("timestamp")).collect()[0]
    logger.info("🚀 Rango desplazado (+8 años): %s  →  %s", rango_desplazado[0], rango_desplazado[1])

except Exception as e:
    logger.error("❌ Error durante el desplazamiento de fechas: %s", e)
    spark.stop()
    sys.exit(1)

# ---------------------------------------------------------------------------
# 9. GUARDAR EN PARQUET (SILVER)
# ---------------------------------------------------------------------------
logger.info("")
logger.info("-" * 70)
logger.info("PASO 8/8: Guardando en formato Parquet (silver/)")
logger.info("-" * 70)

try:
    output_path = os.path.join(SILVER_DIR, f"turbine_{NUMBER_TURBINE}_telemetry_clean.parquet")
    
    # PySpark sobreescribirá la estructura de carpetas de forma nativa sin que salte el IsADirectoryError
    turbine_data_df.write.parquet(output_path, mode="overwrite")

    rel_path = os.path.relpath(output_path, BASE_DIR)
    logger.info("✅ Archivo guardado exitosamente en: ./%s", rel_path)
    logger.info("   📦 Tamaño aproximado: verificar en disco")
    logger.info("📊 NÚMERO TOTAL DE COLUMNAS FINALES: %d", len(turbine_data_df.columns))

except Exception as e:
    logger.error("❌ Error al guardar Parquet: %s", e)
    spark.stop()
    sys.exit(1)

# ---------------------------------------------------------------------------
# 10. LIMPIEZA Y CIERRE
# ---------------------------------------------------------------------------
logger.info("")
logger.info("-" * 70)
logger.info("LIMPIEZA FINAL")
logger.info("-" * 70)

try:
    # Limpiar archivos temporales
    temp_files = glob.glob(os.path.join(TEMP_DIR, "temp_*.csv"))
    for tf in temp_files:
        os.remove(tf)
        logger.info("   🗑️  Eliminado temporal: %s", os.path.basename(tf))

    logger.info("✅ Archivos temporales limpiados.")
except Exception as e:
    logger.warning("⚠️  Error al limpiar temporales: %s", e)

# Detener Spark
try:
    spark.stop()
    logger.info("✅ Spark Session detenida correctamente.")
except Exception as e:
    logger.warning("⚠️  Error al detener Spark: %s", e)

logger.info("")
logger.info("=" * 70)
logger.info("SCRIPT FINALIZADO CON ÉXITO")
logger.info("=" * 70)