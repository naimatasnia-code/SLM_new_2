FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# ── Install CPU-only PyTorch 2.4.1 (satisfies transformers>=2.4 requirement) ──
RUN pip install --no-cache-dir \
    torch==2.4.1 \
    --index-url https://download.pytorch.org/whl/cpu \
    && pip cache purge

# ── Install everything else (torch already pinned, won't be overwritten) ──────
RUN pip install --no-cache-dir -r requirements.txt \
    && pip cache purge

# ── Remove triton if anything sneaked it in ───────────────────────────────────
RUN pip uninstall -y triton 2>/dev/null || true

COPY core ./core
COPY rag ./rag
COPY model ./model
COPY adapters ./adapters
COPY data ./data
COPY api.py .

EXPOSE 8000

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
