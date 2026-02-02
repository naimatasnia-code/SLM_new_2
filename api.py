from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import time, os, shutil, json
from fastapi.concurrency import run_in_threadpool
from model.model_registry import MODEL_REGISTRY
from rag.indexer import build_index, load_documents

from core.component import SLMComponent
from core.domain_component import DomainSLMComponent
from data.doc_to_dataset import build_domain_dataset
from model.domain_trainer import train_domain_lora
from datasets import Dataset

# Paths
UPLOAD_DIR = "uploads"
VECTOR_DIR = "vector_db"
DATASET_PATH = "data/train.jsonl"
LORA_DIR = "lora_adapter"

os.makedirs("data", exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(VECTOR_DIR, exist_ok=True)

# FastAPI
app = FastAPI(title="Loop SLM Orchestrator")

slm_component = None

# Schemas
class ModeRequest(BaseModel):
    mode: str   # "rag" or "finetune"
    model: str  # "phi-2" or "tinyllama"

class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str
    latency_sec: float
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    model: str
    domain_adapter: bool = False

# Upload Files
@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    path = os.path.join(UPLOAD_DIR, file.filename)
    with open(path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Index in threadpool
    await run_in_threadpool(build_index, [path], VECTOR_DIR)

    return {"status": "uploaded", "file": file.filename}

# Setup Mode (RAG or FineTune)
@app.post("/setup")
async def setup(req: ModeRequest):
    global slm_component

    # Resolve model from registry
    model_name = MODEL_REGISTRY.get(req.model, req.model)

    if req.mode not in ["rag", "finetune"]:
        raise HTTPException(400, "mode must be rag or finetune")

    # ---------------- RAG ----------------
    if
