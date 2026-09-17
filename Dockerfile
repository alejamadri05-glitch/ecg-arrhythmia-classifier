# API de clasificación de latidos ECG. Proyecto educativo: no es un dispositivo médico.
#
# El modelo no está en el repositorio (models/ está en .gitignore), así que hay que entrenarlo
# antes de construir la imagen:
#
#   python -m ecg.segment && python -m ecg.train baseline-v5
#   docker build -t ecg-api .
#   docker run -p 8000:8000 ecg-api
#
FROM python:3.12-slim

WORKDIR /app

# libgomp: XGBoost lo necesita para OpenMP
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# La API sirve el baseline (XGBoost) y no necesita PyTorch, que es un extra opcional del paquete.
# En Linux, XGBoost además arrastra NCCL (~290 MB), que solo sirve para entrenar con varias GPU;
# se desinstala porque la inferencia corre en CPU. El nombre cambia con la versión de CUDA
# (nvidia-nccl-cu12, -cu13), por eso se busca por prefijo.
COPY pyproject.toml README.md ./
COPY src/ ./src/
RUN pip install --no-cache-dir ".[api]" \
    && pip freeze | grep -i '^nvidia-nccl' | cut -d= -f1 | xargs -r pip uninstall -y

COPY api/ ./api/
COPY models/ ./models/

ENV ECG_MODEL=/app/models/baseline_v5.joblib
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
