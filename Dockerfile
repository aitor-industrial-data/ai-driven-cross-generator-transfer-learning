# Usamos la imagen oficial de AWS Lambda para Python 3.12
FROM public.ecr.aws/lambda/python:3.12

# Instalamos libgomp usando el gestor ligero nativo de la imagen de Amazon
RUN microdnf install -y libgomp && microdnf clean all

# Copiamos el archivo de requisitos al directorio de trabajo de Lambda
COPY requirements.txt ${LAMBDA_TASK_ROOT}

# Instalamos las dependencias de Python dentro del contenedor
RUN pip install --no-cache-dir -r requirements.txt

# Copiamos el módulo shared con las funciones de feature engineering
COPY src/shared/ ${LAMBDA_TASK_ROOT}/shared/

# Copiamos el script principal de inferencia diaria
COPY src/t2_daily_inference_serverless.py ${LAMBDA_TASK_ROOT}

# Configuración del Handler que invocará AWS Lambda al activarse
CMD [ "t2_daily_inference_serverless.handler" ]