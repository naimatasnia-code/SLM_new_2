

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
MINIO_API_HOST    = os.getenv("MINIO_API_HOST",    "192.168.1.10:9000")
ACCESS_KEY        = os.getenv("ACCESS_KEY",        "minio")
SECRET_KEY        = os.getenv("SECRET_KEY",        "password")
NODE_STORAGE_REF  = os.getenv("NODE_STORAGE_REF",  "nodes_bucket")

MINIO_CLIENT = Minio(
    MINIO_API_HOST,
    access_key=ACCESS_KEY,
    secret_key=SECRET_KEY,
    secure=False,
)

# ── Constants ─────────────────────────────────────────────────────────────────
HAS_GPU        = torch.cuda.is_available()
CPU_SAFE_MODELS = {"tinyllama", "qwen-0.5b"}   # fit in 4 GB RAM on CPU
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
slm_component: SLMComponent | DomainSLMComponent | None = None
current_mode:  str | None = None
current_model: str | None = None


# ── Schemas ───────────────────────────────────────────────────────────────────
class ModeRequest(BaseModel):
    model: str   # "phi-2" | "tinyllama" | "qwen-0.5b"

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
def _validate_model(model_name: str):
    """Raises 400 if model unknown or unsafe for current hardware."""
    if model_name not in ALL_MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model '{model_name}'. Choose from: {sorted(ALL_MODELS)}",
        )
    if not HAS_GPU and model_name not in CPU_SAFE_MODELS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Model '{model_name}' requires a GPU and will exceed 4 GB RAM on CPU. "
                f"Please use one of: {sorted(CPU_SAFE_MODELS)}"
            ),
        )


def _unload_current_model():
    """Frees RAM by deleting the currently loaded model before loading a new one."""
    global slm_component
    if slm_component is not None:
        print("[api] Unloading current model to free RAM...")
        del slm_component
        slm_component = None
        gc.collect()
        if HAS_GPU:
            torch.cuda.empty_cache()
        print("[api] Model unloaded.")


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    """Reports system status, RAM, GPU, and loaded model."""
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
    path = os.path.join(UPLOAD_DIR, file.filename)
    with open(path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    await run_in_threadpool(build_index, [path], VECTOR_DIR)

    return {"node": "upload", "status": "uploaded", "file": file.filename}


@app.post("/node/rag")
async def rag_node(req: ModeRequest):
    global slm_component, current_mode, current_model

    _validate_model(req.model)       # blocks phi-2 on CPU
    _unload_current_model()          # free RAM before loading new model

    slm_component = await run_in_threadpool(
        lambda: SLMComponent(model_name=req.model, vector_dir=VECTOR_DIR)
    )
    current_mode  = "rag"
    current_model = req.model

    return {"node": "rag", "status": "ready", "model": req.model}


@app.post("/node/finetune")
async def finetune_node(req: ModeRequest):
    global slm_component, current_mode, current_model

    _validate_model(req.model)

    # ── Step 1: unload inference model FIRST → frees RAM for training ─────────
    _unload_current_model()

    # ── Step 2: build dataset ─────────────────────────────────────────────────
    file_paths = [os.path.join(UPLOAD_DIR, f) for f in os.listdir(UPLOAD_DIR)]
    docs = await run_in_threadpool(load_documents, file_paths)
    await run_in_threadpool(build_domain_dataset, docs, DATASET_PATH)

    with open(DATASET_PATH) as f:
        data = [json.loads(line) for line in f]
    if not data:
        raise HTTPException(500, "Dataset is empty. No training samples generated.")

    dataset = Dataset.from_list(data)

    # ── Step 3: train LoRA (training model loads, trains, then self-unloads) ──
    try:
        await run_in_threadpool(train_domain_lora, req.model, dataset, LORA_DIR)
    except MemoryError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # ── Step 4: reload inference model with LoRA adapter ─────────────────────
    slm_component = await run_in_threadpool(
        lambda: DomainSLMComponent(req.model, VECTOR_DIR, LORA_DIR)
    )
    current_mode  = "finetune"
    current_model = req.model

    return {"node": "finetune", "status": "ready", "model": req.model}


@app.post("/node/inference", response_model=QueryResponse)
async def inference_node(req: QueryRequest):
    global slm_component

    if slm_component is None:
        print(f"[api] slm_component is None at {datetime.datetime.now()}")
        await asyncio.sleep(5)
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
    return await inference_node(req)
