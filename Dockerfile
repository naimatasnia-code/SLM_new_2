FROM python:3.11-slim

WORKDIR /app

# Install only essential system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# ── KEY FIX: Install CPU-only PyTorch FIRST before everything else ────────────
# This prevents pip from pulling in the full CUDA build (~2.5GB saved)
# and avoids triton/cupti GPU libs that cause "no space left" errors
RUN pip install --no-cache-dir \
    torch==2.2.2 \
    --index-url https://download.pytorch.org/whl/cpu \
    && pip cache purge

# Now install the rest of requirements (torch already satisfied, won't re-download)
RUN pip install --no-cache-dir -r requirements.txt \
    && pip cache purge

# Remove triton if it still got pulled in (GPU-only, useless on CPU)
RUN pip uninstall -y triton 2>/dev/null || true

COPY core ./core
COPY rag ./rag
COPY model ./model
COPY adapters ./adapters
COPY data ./data
COPY api.py .

EXPOSE 8000

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
