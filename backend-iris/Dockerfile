FROM python:3.11-slim

WORKDIR /app

# Instalacion de curl para habilitar el healthcheck
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Copia de dependencias para optimizacion de la cache de capas
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia del codigo fuente de la API
COPY . .

EXPOSE 8000

# Configuracion del arranque ordenado y estado healthy
HEALTHCHECK --interval=10s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]