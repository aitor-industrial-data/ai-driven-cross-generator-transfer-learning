import os
import requests

# Definir la ruta exacta solicitada
target_dir = "/home/aitor/Documentos/ai-driven-cross-generator-transfer-learning/data/bronze"
os.makedirs(target_dir, exist_ok=True)

# ID del repositorio de Kelmarsh en Zenodo (v4 hasta 2024)
record_id = "16807551"
api_url = f"https://zenodo.org/api/records/{record_id}"

print(f"📡 Solicitando metadatos a Zenodo para el registro: {record_id}...")
response = requests.get(api_url)

if response.status_code == 200:
    files = response.json().get("files", [])
    print(f"📦 Se han encontrado {len(files)} archivos para descargar.\n")
    
    for file_info in files:
        filename = file_info.get("key")
        # Filtrar para descargar solo los SCADA y los archivos de logs de estado/fallos
        if "SCADA" in filename or "status" in filename or "log" in filename:
            download_url = file_info.get("links", {}).get("self")
            file_path = os.path.join(target_dir, filename)
            
            print(f"⬇️ Descargando: {filename}...")
            with requests.get(download_url, stream=True) as r:
                r.raise_for_status()
                with open(file_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
            print(f"✅ Guardado en: {file_path}\n")
            
    print("🚀 ¡Descarga completa de todos los archivos del parque en la carpeta bronze!")
else:
    print(f"❌ Error al conectar con Zenodo. Código de estado: {response.status_code}")