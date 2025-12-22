# ====== BUILDER STAGE ======
FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    POETRY_VIRTUALENVS_CREATE=false \
    POETRY_NO_INTERACTION=1 \
    DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libglib2.0-0 \
    libopenblas-dev \
    libomp5 \
    libblas-dev \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip setuptools wheel

# Install Poetry
RUN pip install --no-cache-dir poetry==2.2.0 poetry-plugin-export

# Export dependencies
COPY pyproject.toml poetry.lock ./

RUN poetry export \
    -f requirements.txt \
    --output requirements.txt \
    --without-hashes \
    --only main

# Install dependencies into unified target
RUN pip install \
    --no-cache-dir \
    --target=/install/python \
    -r requirements.txt \
    && pip install \
    --no-cache-dir \
    --target=/install/python \
    typing-extensions

# ====== RUNTIME STAGE ======
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Runtime system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libopenblas0 \
    libomp5 \
    libblas3 \
    && rm -rf /var/lib/apt/lists/*

# Copy dependencies
COPY --from=builder /install/python \
    /usr/local/lib/python3.11/site-packages

COPY app ./app
COPY frontend ./frontend

RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--access-log"]
