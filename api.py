from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import time, os, shutil, json, torch
from fastapi.concurrency import run_in_threadpool
from typing import Optional

# Internal modules
from model.model_registry import MODEL_REGISTRY
from rag.indexer import build_index, load_documents
from core.component import SLMComponent
from core.domain_component import DomainSLMComponent
from data.doc_to_dataset import build_domain_dataset
from model.domain_trainer import train_domain_lora
from datasets import Dataset

# ---------------- Paths ----------------
UPLOAD_DIR = "uploads"
VECTOR_DIR = "vector_db"
DATASET_PATH = "data/train.jsonl"
LORA_DIR = "lora_adapter"

os.makedirs("data", exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(VECTOR_DIR, exist_ok=True)

# ---------------- FastAPI ----------------
app = FastAPI(title="SLM Component")

slm_component = None
current_mode = None
current_model = None

# ---------------- Schemas ----------------
class ModeRequest(BaseModel):
    mode: str   # "rag" or "finetune"
    model: str  # "tinyllama" | "phi-2"

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


# =====================================================
# Upload Endpoint
# =====================================================
@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    path = os.path.join(UPLOAD_DIR, file.filename)
    with open(path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Build vector DB in background thread
    await run_in_threadpool(build_index, [path], VECTOR_DIR)

    return {"status": "uploaded", "file": file.filename}


# =====================================================
# Setup Endpoint (RAG or Fine-tune)
# =====================================================
@app.post("/setup")
async def setup(req: ModeRequest):
    global slm_component, current_mode, current_model

    if req.mode not in ["rag", "finetune"]:
        raise HTTPException(400, "mode must be 'rag' or 'finetune'")

    # Resolve HF model name
    model_name = MODEL_REGISTRY.get(req.model, req.model)

    # Detect GPU
    has_gpu = torch.cuda.is_available()
    print(f"GPU available: {has_gpu}")

    # ---------------- RAG MODE ----------------
    if req.mode == "rag":
        slm_component = SLMComponent(model_name=model_name, vector_dir=VECTOR_DIR)
        current_mode = "rag"
        current_model = model_name

        return {"status": "ready", "mode": "rag", "model": model_name}


    # ---------------- FINE-TUNE MODE ----------------
    if req.mode == "finetune":

        # Safety: prevent CPU fine-tuning large models
        if (not has_gpu) and ("phi-2" in model_name):
            raise HTTPException(
                400,
                "Fine-tuning phi-2 on CPU is disabled. Use tinyllama or GPU."
            )

        # Collect uploaded files
        files = [os.path.join(UPLOAD_DIR, f) for f in os.listdir(UPLOAD_DIR)]
        if not files:
            raise HTTPException(400, "No uploaded documents found.")

        # Load documents
        docs = await run_in_threadpool(load_documents, files)

        # Build dataset
        await run_in_threadpool(build_domain_dataset, docs, DATASET_PATH)

        # Load dataset JSONL
        with open(DATASET_PATH) as f:
            data = [json.loads(l) for l in f]
        dataset = Dataset.from_list(data)

        # Train LoRA adapter (threadpool)
        await run_in_threadpool(train_domain_lora, model_name, dataset, LORA_DIR)

        # Load adapted model
        slm_component = DomainSLMComponent(model_name, VECTOR_DIR, LORA_DIR)

        current_mode = "finetune"
        current_model = model_name

        return {"status": "ready", "mode": "finetune", "model": model_name}


# =====================================================
# Chat Endpoint
# =====================================================
@app.post("/chat", response_model=QueryResponse)
async def chat(req: QueryRequest):
    global slm_component

    if slm_component is None:
        raise HTTPException(400, "Model not initialized. Call /setup first.")

    start = time.time()

    result = await run_in_threadpool(slm_component.run, req.question)

    latency = round(time.time() - start, 3)
    result["latency_sec"] = latency

    # Add mode flag for dashboard
    result["domain_adapter"] = (current_mode == "finetune")

    # Log for Loop metrics
    print(f"[METRIC] model={current_model} mode={current_mode} latency={latency}s tokens={result['total_tokens']}")

    return result
