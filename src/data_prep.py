import os
import re
import glob



# ==============================================================================
# 3.1 Limpieza de Headers de los CSV Bronze
# ==============================================================================
def clean_header_line(line):
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


def get_cleaned_files_info(target_years, turbine_number):
    """
    Devuelve lista de diccionarios con info de archivos limpios.
    
    Args:
        target_years: Lista de años (ej: ["2018", "2019", "2020"])
        turbine_number: Número de turbina (default: 1)
    
    Returns:
        Lista de dicts: {'path', 'filename', 'header', 'skip_lines'}
    """
    base_dir = os.path.dirname(os.getcwd())
    bronze_dir = os.path.join(base_dir, "data", "bronze")
    all_files = []
    for year in target_years:
        pattern = os.path.join(
            bronze_dir, 
            f"Kelmarsh_SCADA_{year}_*", 
            f"Turbine_Data_Kelmarsh_{turbine_number}_*.csv"
        )
        all_files.extend(sorted(glob.glob(pattern)))
    
    print(f"📁 Archivos encontrados: {len(all_files)}")
    
    files_info = []
    
    for file_path in all_files:
        filename = os.path.basename(file_path)
        print(f"\n🔄 Analizando: {filename}")
        
        # Leer solo las primeras 10 líneas para encontrar el header
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_number, line in enumerate(f, 1):
                if line_number == 10 and 'Date and time' in line:
                    cleaned_header = clean_header_line(line)
                    files_info.append({
                        'path': file_path,
                        'filename': filename,
                        'header': cleaned_header,
                        'skip_lines': 9  # saltar las 9 primeras (comentarios)
                    })
                    print(f"   ✅ Header limpio: {cleaned_header[:80]}...")
                    break
    
    print(f"\n✅ Total de archivos listos: {len(files_info)}")
    return files_info


# ==============================================================================
# 3.2 Carga de Telemetría con Spark
# ==============================================================================

import os
import tempfile
from typing import List, Dict

from pyspark.sql import SparkSession, DataFrame
from IPython.display import display, HTML


def load_telemetry_with_spark(files_info: List[Dict],
                              spark: SparkSession = None,
                              temp_dir: str = None,
                              app_name: str = "Kelmarsh-EDA-Notebook",
                              master: str = "local[6]",
                              driver_memory: str = "8g") -> DataFrame:
    """
    Carga archivos CSV de telemetría limpios con Spark, los une en un DataFrame.
    
    Args:
        files_info: Lista de dicts con 'path', 'filename', 'header', 'skip_lines'
        spark: SparkSession existente (None = crea nueva)
        temp_dir: Directorio para archivos temporales (None = auto)
        app_name: Nombre de la app Spark (si crea sesión)
        master: Master URL (si crea sesión)
        driver_memory: Memoria del driver (si crea sesión)
    
    Returns:
        DataFrame unificado con todos los archivos
    """
    
    
    # 1. SPARK SESSION (crear si no se pasó)
    if spark is None:
        spark = SparkSession.builder \
            .appName(app_name) \
            .master(master) \
            .config("spark.driver.memory", driver_memory) \
            .getOrCreate()
        print(f"✅ SparkSession creada: {app_name} | {master} | {driver_memory}")
    else:
        print(f"✅ Usando SparkSession existente")
    
   
    # 2. VERIFICAR files_info
    if not files_info:
        raise ValueError("❌ 'files_info' está vacío. Ejecuta get_cleaned_files_info() primero.")
    
    print(f"📁 Archivos a cargar: {len(files_info)}")
    for info in files_info:
        print(f"   - {info['filename']}")
    
   
    # 3. PREPARAR DIRECTORIO TEMPORAL
    if temp_dir is None:
        temp_dir = os.path.join(os.path.dirname(os.getcwd()), "data", "temp_spark")
    os.makedirs(temp_dir, exist_ok=True)
    
   
    # 4. CARGAR CADA ARCHIVO
    spark_dfs = []
    
    for info in files_info:
        file_path = info['path']
        header = info['header']
        skip_lines = info.get('skip_lines', 9)
        filename = info['filename']
        
        print(f"\n🔹 Cargando: {filename}")
        
        # Leer archivo original, saltar líneas de comentarios + header original
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()[skip_lines + 1:]
        
        # Insertar header limpio
        lines.insert(0, header + '\n')
        
        # Escribir temporal limpio
        temp_file = os.path.join(temp_dir, f"temp_{filename}")
        with open(temp_file, 'w', encoding='utf-8') as f:
            f.writelines(lines)
        
        # Leer con Spark
        sdf = spark.read \
            .option("header", "true") \
            .option("inferSchema", "true") \
            .csv(temp_file)
        
        spark_dfs.append(sdf)
        print(f"   ✅ {sdf.count()} filas x {len(sdf.columns)} columnas")
    

    # 5. UNIR DATAFRAMES
    if len(spark_dfs) == 1:
        turbine_data_df = spark_dfs[0]
    else:
        turbine_data_df = spark_dfs[0]
        for sdf in spark_dfs[1:]:
            turbine_data_df = turbine_data_df.unionByName(sdf, allowMissingColumns=True)
    
  
    # 6. RESULTADOS
    total_records = turbine_data_df.count()
    
    print(f"\n🎯 Total de registros: {total_records}")
    print(f"📊 NÚMERO TOTAL DE COLUMNAS: {len(turbine_data_df.columns)}")
    
    return turbine_data_df


def preview_df(df: DataFrame, n: int = 5) -> None:
    """
    Muestra preview del DataFrame en formato HTML ancho.
    """
    pdf = df.limit(n).toPandas()
    display(HTML(
        f'<div style="max-width:100%;overflow-x:auto;font-size:11px;white-space:nowrap">'
        f'{pdf.to_html(index=False)}</div>'
    ))