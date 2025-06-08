# Usa una imagen oficial de Python como base
FROM python:3.11-slim

# Instala dependencias del sistema necesarias para EasyOCR
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Establece el directorio de trabajo
WORKDIR /app

# Copia requirements.txt y lo instala
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Copia el resto del código
COPY . .

# Expone el puerto 8080 (opcional pero recomendado)
EXPOSE 8080

# Railway inyecta automáticamente la variable de entorno PORT
# uvicorn usará el puerto que se le pase desde Railway.
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8080"]
