# Usa una imagen oficial de Python como base
FROM python:3.11-slim

# Instala dependencias del sistema necesarias para EasyOCR y PaddleOCR
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Establece el directorio de trabajo
WORKDIR /app

# Copia requirements.txt sin paddlepaddle
COPY requirements.txt .

# Instala paddlepaddle CPU desde índice oficial chino
RUN pip install --no-cache-dir paddlepaddle==3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/

# Instala el resto de dependencias
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Copia el resto del código
COPY . .

# Expone el puerto 8080
EXPOSE 8080

# Comando para iniciar la app
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8080"]
