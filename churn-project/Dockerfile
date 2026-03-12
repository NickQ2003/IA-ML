# ─────────────────────────────────────────────
# Stage 1: Entrenamiento del modelo
# ─────────────────────────────────────────────
FROM python:3.11-slim AS trainer

WORKDIR /build

# Instalar dependencias primero (capa cacheada por Docker)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código de entrenamiento y entrenar el modelo
COPY train_model.py .
RUN mkdir -p app && python train_model.py

# ─────────────────────────────────────────────
# Stage 2: Imagen de producción (más liviana)
# ─────────────────────────────────────────────
FROM python:3.11-slim AS production

WORKDIR /app

# Instalar solo dependencias de runtime
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar la app y el modelo ya entrenado desde el stage anterior
COPY app/ .
COPY --from=trainer /build/app/model.pkl ./model.pkl

# Puerto de la API
EXPOSE 5000

# Gunicorn: servidor WSGI de producción (4 workers)
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "4", "--timeout", "120", "app:app"]