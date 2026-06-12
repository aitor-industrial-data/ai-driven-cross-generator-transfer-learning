import os
import glob
import shutil
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

# 1. SPARK
spark = SparkSession.builder \
    .appName("Kelmarsh-Fault-Prepare") \
    .master("local[6]") \
    .config("spark.driver.memory", "8g") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

# 2. RUTAS
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
bronze_dir = os.path.join(PROJECT_ROOT, "data", "bronze")

# 3. ARCHIVOS
target_years = ["2018", "2019", "2020", "2021", "2022"]
TURBINE_ID = 2

all_files = []
for year in target_years:
    pattern = os.path.join(bronze_dir, f"Kelmarsh_SCADA_{year}_*", f"Status_Kelmarsh_{TURBINE_ID}_*.csv")
    all_files.extend(glob.glob(pattern))

print(f"Archivos: {len(all_files)}")

# 4. LEER CSV
status_df = spark.read.option("header", True).option("comment", "#").csv(all_files)

# 5. SUMAR 8 SOLO AL AÑO
# Extraer año, sumar 8, reconstruir el string completo
status_df = status_df.withColumn(
    "Timestamp start",
    F.when(
        F.col("Timestamp start") == "-",
        F.col("Timestamp start")
    ).otherwise(
        F.concat(
            (F.substring("Timestamp start", 1, 4).cast("int") + 8).cast("string"),
            F.substring("Timestamp start", 5, 100)
        )
    )
).withColumn(
    "Timestamp end",
    F.when(
        F.col("Timestamp end") == "-",
        F.col("Timestamp end")
    ).otherwise(
        F.concat(
            (F.substring("Timestamp end", 1, 4).cast("int") + 8).cast("string"),
            F.substring("Timestamp end", 5, 100)
        )
    )
)

# 6. ORDENAR
status_df = status_df.withColumn(
    "_sort_ts",
    F.try_to_timestamp(F.col("Timestamp start"))
).orderBy("_sort_ts").drop("_sort_ts")

# 7. CREAR DIRECTORIO DE SALIDA
output_dir = os.path.join(PROJECT_ROOT, "data", "silver")
os.makedirs(output_dir, exist_ok=True)

output_file = os.path.join(output_dir, "turbine_2_status_2026_2030.csv")
temp_dir = os.path.join(output_dir, "temp_turbine_2")

# 8. GUARDAR EN UN SOLO CSV
status_df.coalesce(1).write.mode("overwrite").option("header", "true").csv(temp_dir)

# 9. RENOMBRAR EL ARCHIVO part-00000 AL NOMBRE FINAL
for f in os.listdir(temp_dir):
    if f.startswith("part-") and f.endswith(".csv"):
        shutil.move(os.path.join(temp_dir, f), output_file)
        break

shutil.rmtree(temp_dir)

count = status_df.count()
print(f"\nGuardado: {output_file} ({count} registros)")
print(f"Listo. Archivo único generado en: {output_file}")