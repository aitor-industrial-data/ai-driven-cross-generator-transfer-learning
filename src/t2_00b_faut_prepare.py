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

# 5. SUMAR 8 AÑOS
status_df = status_df.withColumn(
    "Timestamp start",
    F.add_months(F.try_to_timestamp("Timestamp start"), 96)
).withColumn(
    "Timestamp end",
    F.add_months(F.try_to_timestamp("Timestamp end"), 96)
)

# 6. AÑADIR COLUMNA AÑO-MES PARA FRACCIONAR
status_df = status_df.withColumn("year_month", F.date_format("Timestamp start", "yyyy-MM"))

# 7. OBTENER LISTA DE MESES ÚNICOS
months = [row.year_month for row in status_df.select("year_month").distinct().collect()]
months.sort()

# 8. CREAR DIRECTORIO DE SALIDA
output_dir = os.path.join(PROJECT_ROOT, "data", "silver", "turbine_2_status_by_month")
os.makedirs(output_dir, exist_ok=True)

# 9. GUARDAR UN CSV POR MES
for month in months:
    month_df = status_df.filter(F.col("year_month") == month).drop("year_month")
    month_df = month_df.orderBy("Timestamp start")
    
    output_file = os.path.join(output_dir, f"turbine_2_status_{month}.csv")
    temp_dir = os.path.join(output_dir, f"temp_{month}")
    
    month_df.coalesce(1).write.mode("overwrite").option("header", "true").csv(temp_dir)
    
    # Renombrar el archivo part-00000
    for f in os.listdir(temp_dir):
        if f.startswith("part-") and f.endswith(".csv"):
            shutil.move(os.path.join(temp_dir, f), output_file)
            break
    shutil.rmtree(temp_dir)
    
    count = month_df.count()
    print(f"Guardado: {output_file} ({count} registros)")

print(f"\nListo. Archivos guardados en: {output_dir}")