FROM python:3.11-slim

WORKDIR /app

# System deps (FAISS needs this)
RUN apt-get update && apt-get install -y \
    build-essential \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml poetry.lock ./

RUN pip install poetry \
 && poetry config virtualenvs.create false \
 && poetry install --no-interaction --no-ansi

COPY app ./app
COPY frontend ./frontend

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
