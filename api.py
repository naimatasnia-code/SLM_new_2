

import gc
import os
import time
import json
import shutil
import asyncio
import datetime

import torch
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datasets import Dataset
from minio import Minio

from rag.indexer import build_index, load_documents
from core.component import SLMComponent
from core.domain_component import DomainSLMComponent
from data.doc_to_dataset import build_domain_dataset
from model.domain_trainer import train_domain_lora

# ── Paths ─────────────────────────────────────────────────────────────────────
UPLOAD_DIR   = "uploads"
VECTOR_DIR   = "vector_db"
DATASET_PATH = "data/train.jsonl"
LORA_DIR     = "lora_adapter"

os.makedirs("data",      exist_ok=True)
os.makedirs(UPLOAD_DIR,  exist_ok=True)
os.makedirs(VECTOR_DIR,  exist_ok=True)

# ── MinIO ─────────────────────────────────────────────────────────────────────
MINIO_API_HOST   = os.getenv("MINIO_API_HOST",   "192.168.1.10:9000")
ACCESS_KEY       = os.getenv("ACCESS_KEY",       "minio")
SECRET_KEY       = os.getenv("SECRET_KEY",       "password")
NODE_STORAGE_REF = os.getenv("NODE_STORAGE_REF", "nodes_bucket")

MINIO_CLIENT = Minio(
    MINIO_API_HOST,
    access_key=ACCESS_KEY,
    secret_key=SECRET_KEY,
    secure=False,
)

# ── Hardware constants ────────────────────────────────────────────────────────
HAS_GPU         = torch.cuda.is_available()
CPU_SAFE_MODELS = {"tinyllama", "qwen-0.5b"}
ALL_MODELS      = {"phi-2", "tinyllama", "qwen-0.5b"}

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="Chatbot SLM Component")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# ── Global state ──────────────────────────────────────────────────────────────
slm_component = None
current_mode  = None
current_model = None


# ── Schemas ───────────────────────────────────────────────────────────────────
class ModeRequest(BaseModel):
    model: str

class FineTuneRequest(BaseModel):
    model: str
    epochs: int = 1
    samples_per_chunk: int = 2

class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer:            str
    latency_sec:       float
    prompt_tokens:     int
    completion_tokens: int
    total_tokens:      int
    model:             str
    domain_adapter:    bool = False


# ── Helpers ───────────────────────────────────────────────────────────────────
def _validate_model(model_name: str) -> None:
    if model_name not in ALL_MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model '{model_name}'. Choose from: {sorted(ALL_MODELS)}",
        )
    if not HAS_GPU and model_name not in CPU_SAFE_MODELS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Model '{model_name}' requires ~6 GB RAM on CPU. "
                f"For 4 GB environments use: {sorted(CPU_SAFE_MODELS)}"
            ),
        )


def _unload_current_model() -> None:
    global slm_component
    if slm_component is not None:
        print("[api] Unloading current model...")
        del slm_component
        slm_component = None
        gc.collect()
        if HAS_GPU:
            torch.cuda.empty_cache()
        print("[api] Model unloaded. RAM freed.")


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    import psutil
    mem = psutil.virtual_memory()
    return {
        "status":        "ok",
        "gpu_available": HAS_GPU,
        "ram_total_gb":  round(mem.total / 1e9, 2),
        "ram_used_gb":   round(mem.used  / 1e9, 2),
        "ram_free_gb":   round(mem.available / 1e9, 2),
        "model_loaded":  current_model,
        "mode":          current_mode,
        "timestamp":     datetime.datetime.utcnow().isoformat(),
    }


@app.post("/node/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload a PDF or DOCX and index it into the vector DB."""
    path = os.path.join(UPLOAD_DIR, file.filename)
    with open(path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    await run_in_threadpool(build_index, [path], VECTOR_DIR)
    return {"node": "upload", "status": "indexed", "file": file.filename}


@app.post("/node/rag")
async def rag_node(req: ModeRequest):
    """Load model in RAG-only mode (no fine-tuning)."""
    global slm_component, current_mode, current_model

    _validate_model(req.model)
    _unload_current_model()

    slm_component = await run_in_threadpool(
        lambda: SLMComponent(model_name=req.model, vector_dir=VECTOR_DIR, top_k=5)
    )
    current_mode  = "rag"
    current_model = req.model
    return {"node": "rag", "status": "ready", "model": req.model}


@app.post("/node/finetune")
async def finetune_node(req: FineTuneRequest):
    """
    Fine-tune a LoRA adapter on uploaded documents, then reload in RAG mode.

    Body params:
      model             : "qwen-0.5b" | "tinyllama"  (phi-2 blocked on CPU)
      epochs            : training epochs (default: 1, keep low for 4 GB)
      samples_per_chunk : Q&A pairs per document chunk (default: 2)

    Prerequisites:
      - Upload at least one document via POST /node/upload first.

    Returns:
      training_samples : number of examples the model was trained on
      adapter_path     : where the LoRA adapter was saved
    """
    global slm_component, current_mode, current_model

    _validate_model(req.model)

    # ── Step 1: free RAM ──────────────────────────────────────────────────────
    _unload_current_model()

    # ── Step 2: collect uploaded files ───────────────────────────────────────
    file_paths = [
        os.path.join(UPLOAD_DIR, f)
        for f in os.listdir(UPLOAD_DIR)
        if f.lower().endswith((".pdf", ".docx"))
    ]
    if not file_paths:
        raise HTTPException(
            status_code=400,
            detail="No documents found. Upload at least one PDF or DOCX via /node/upload first.",
        )

    # ── Step 3: load document text ────────────────────────────────────────────
    docs = await run_in_threadpool(load_documents, file_paths)
    if not docs:
        raise HTTPException(
            status_code=400,
            detail="Could not extract text from uploaded documents.",
        )

    # ── Step 4: build training dataset ───────────────────────────────────────
    # Creates instruction-following Q&A pairs + negative (out-of-scope) samples
    n_samples = await run_in_threadpool(
        build_domain_dataset,
        docs,
        DATASET_PATH,
        80,                     # min_chunk_len — skip noise/header chunks
        req.samples_per_chunk,  # Q&A pairs per chunk
        0.15,                   # 15% out-of-scope negative samples
    )
    if n_samples == 0:
        raise HTTPException(
            status_code=500,
            detail="Dataset generation produced 0 samples. Check document content.",
        )
    print(f"[api] Built {n_samples} training samples from {len(file_paths)} file(s).")

    with open(DATASET_PATH) as f:
        data = [json.loads(line) for line in f if line.strip()]

    dataset = Dataset.from_list(data)

    # ── Step 5: train LoRA adapter ────────────────────────────────────────────
    # domain_trainer loads model → trains → saves adapter → self-destructs
    # So only one model in RAM at a time (critical for 4 GB)
    try:
        await run_in_threadpool(train_domain_lora, req.model, dataset, LORA_DIR)
    except MemoryError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")

    # ── Step 6: reload inference model with adapter ───────────────────────────
    # RAM is now free (trainer self-destructed after Step 5)
    try:
        slm_component = await run_in_threadpool(
            lambda: DomainSLMComponent(req.model, VECTOR_DIR, LORA_DIR)
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Fine-tune succeeded but failed to reload model: {str(e)}",
        )

    current_mode  = "finetune"
    current_model = req.model

    return {
        "node":             "finetune",
        "status":           "ready",
        "model":            req.model,
        "training_samples": n_samples,
        "adapter_path":     LORA_DIR,
    }


@app.post("/node/inference", response_model=QueryResponse)
async def inference_node(req: QueryRequest):
    """Query the currently loaded model."""
    global slm_component

    if slm_component is None:
        await asyncio.sleep(3)
        if slm_component is None:
            raise HTTPException(
                status_code=400,
                detail="No model loaded. Call /node/rag or /node/finetune first.",
            )

    start  = time.time()
    result = await run_in_threadpool(slm_component.run, req.question)
    result["latency_sec"]    = round(time.time() - start, 3)
    result["domain_adapter"] = (current_mode == "finetune")
    return result


@app.post("/chat", response_model=QueryResponse)
async def chat(req: QueryRequest):
    """Alias for /node/inference."""
    return await inference_node(req)
