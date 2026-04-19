# ── Web framework ─────────────────────────────────────────────────────────────
fastapi
uvicorn
python-multipart

# ── LangChain ─────────────────────────────────────────────────────────────────
langchain
langchain-community
langchain-huggingface
langchain-text-splitters

# ── ML / AI ───────────────────────────────────────────────────────────────────
# torch is intentionally NOT listed here.
# It is installed separately in Dockerfile with CPU-only wheels.
transformers==4.46.3
sentence-transformers==3.3.1
datasets
peft
accelerate

# ── NumPy pinned to <2 to avoid binary compatibility crashes ──────────────────
numpy<2

# ── Vector DB / RAG ───────────────────────────────────────────────────────────
faiss-cpu
chromadb
pysqlite3-binary

# ── Document parsing ──────────────────────────────────────────────────────────
pypdf
unstructured
python-docx

# ── Storage / Utils ───────────────────────────────────────────────────────────
minio>=7.2
psutil
