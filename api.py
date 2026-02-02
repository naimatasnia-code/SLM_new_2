from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import time, os, shutil, json
from fastapi.concurrency import run_in_threadpool

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

# 
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

    if req.mode not in ["rag", "finetune"]:
        raise HTTPException(400, "mode must be rag or finetune")

    # RAG
    if req.mode == "rag":
        slm_component = SLMComponent(model_name=req.model, vector_dir=VECTOR_DIR)
        return {"status": "ready", "mode": "rag", "model": req.model}

    # Finetune
    if req.mode == "finetune":

        # Load docs
        docs = await run_in_threadpool(load_documents, [UPLOAD_DIR])

        # Build dataset
        await run_in_threadpool(build_domain_dataset, docs, DATASET_PATH)

        # Load dataset JSONL
        with open(DATASET_PATH) as f:
            data = [json.loads(l) for l in f]
        dataset = Dataset.from_list(data)

        # Train LoRA (CPU optimized)
        await run_in_threadpool(train_domain_lora, req.model, dataset, LORA_DIR)

        # Load domain adapted model
        slm_component = DomainSLMComponent(req.model, VECTOR_DIR, LORA_DIR)

        return {"status": "ready", "mode": "finetune", "model": req.model}

# 
# Chat Endpoint
# 
@app.post("/chat", response_model=QueryResponse)
async def chat(req: QueryRequest):
    global slm_component

    if slm_component is None:
        raise HTTPException(400, "Model not initialized. Call /setup first.")

    start = time.time()

    result = await run_in_threadpool(slm_component.run, req.question)

    result["latency_sec"] = round(time.time() - start, 3)
    return result
