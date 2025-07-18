FROM python:3.11-slim

WORKDIR /app

# System dependencies for TensorFlow
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Increase timeout and use official PyPI mirror
RUN pip install --default-timeout=100 --no-cache-dir -r requirements.txt

RUN pip install watchgod





COPY src/ src/
COPY tomate_model.keras .

EXPOSE 8000

CMD ["uvicorn", "src.infer:app", "--host", "0.0.0.0", "--port", "8000"]
