FROM python:3.11-slim

WORKDIR /app

ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# System dependencies for OpenCV, PyMuPDF, and health-check curl
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application source — preserve the package layout so Path(__file__).parent
# resolves correctly inside Api/main.py
COPY config.yaml .
COPY settings.py .
COPY store.py .
COPY Api/ Api/
COPY Worker/ Worker/

# Writable directories. DATA_DIR defaults to /app; on Railway set it to the
# mount path of an attached volume (e.g. /data) so these survive redeploys.
RUN mkdir -p /app/uploads /app/outputs /app/tmp-work

EXPOSE 8080

# $PORT is injected by Railway; the fallback keeps `docker run` working.
CMD ["sh", "-c", "exec uvicorn Api.main:app --host 0.0.0.0 --port ${PORT:-8080} --proxy-headers --forwarded-allow-ips '*' --timeout-keep-alive 120"]
