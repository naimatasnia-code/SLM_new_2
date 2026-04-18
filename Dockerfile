FROM python:3.11-slim

WORKDIR /app

# Install only essential system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for layer caching
COPY requirements.txt .

# Install dependencies, skip cache to save disk space
RUN pip install --no-cache-dir -r requirements.txt \
    && pip cache purge

# Copy source code only (NO model weights — they come via volume or HF cache)
COPY core ./core
COPY rag ./rag
COPY model ./model
COPY adapters ./adapters
COPY data ./data
COPY api.py .

# Expose port
EXPOSE 8000

# Run FastAPI
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
