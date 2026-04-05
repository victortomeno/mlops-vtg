FROM python:3.11-slim

WORKDIR /app

# Copia de dependencias para optimizacion de la cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia del codigo fuente del frontend
COPY . .

# Puerto estandar de Gradio
EXPOSE 7860

# Variable de entorno para la conexion con el backend
ENV BACKEND_URL=http://localhost:8000

# Arranque de la aplicacion
CMD ["python", "gradio_app.py"]