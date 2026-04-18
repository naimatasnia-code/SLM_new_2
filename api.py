import gc
import os
import re
import time
import json
import shutil
import asyncio
import datetime
import traceback

import torch
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
from datasets import Dataset
from minio import Minio

from rag.indexer import build_index, load_documents
from core.component import SLMComponent
from core.bio_component import BioSLMComponent
from core.domain_component import DomainSLMComponent
from data.doc_to_dataset import build_domain_dataset
from model.domain_trainer import train_domain_lora

import tempfile
import os, sys
import logging
import logging.config

LOGGING_DEFAULT_DICT = {
    "version": 1,
    "disable_existing_loggers": True,
    "formatters": {
        "detailed": {
            "class": "logging.Formatter",
            "format": '"[%(asctime)s] [%(levelname)s] [%(funcName)s():%(lineno)s] '
                      '[PID:%(process)d TID:%(thread)d] %(message)s"',
        }
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": "DEBUG",
            "formatter": "detailed",
            "stream": sys.stdout,
        },
    },
    "loggers": {
        "welcome.log": {"level": "DEBUG", "handlers": ["console"], "propagate": False},
    },
    "root": {"level": "INFO", "handlers": ["console"]},
}

logging.config.dictConfig(LOGGING_DEFAULT_DICT)
log = logging.getLogger("welcome.log")


# ── Paths ─────────────────────────────────────────────────────────────────────
UPLOAD_DIR   = "uploads"
VECTOR_DIR   = "vector_db"
DATASET_PATH = "data/train.jsonl"
ADAPTERS_DIR = "adapters"

os.makedirs("data",       exist_ok=True)
os.makedirs(UPLOAD_DIR,   exist_ok=True)
os.makedirs(VECTOR_DIR,   exist_ok=True)
os.makedirs(ADAPTERS_DIR, exist_ok=True)

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

MODEL_CATALOG = [
    {"id": "tinyllama",  "label": "TinyLlama (1.1B)",  "cpu_safe": True,  "vram_gb": 1.5},
    {"id": "qwen-0.5b",  "label": "Qwen 0.5B",         "cpu_safe": True,  "vram_gb": 1.0},
    {"id": "phi-2",      "label": "Phi-2 (2.7B)",       "cpu_safe": False, "vram_gb": 6.0},
]

# ── Global state ──────────────────────────────────────────────────────────────
slm_component = None
current_mode  = None
current_model = None


# ── Auto-load on startup ──────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global slm_component, current_mode, current_model

    default_model        = os.getenv("DEFAULT_MODEL",        "tinyllama")
    default_adapter_path = os.getenv("DEFAULT_ADAPTER_PATH", "adapters/derma_v1.0")
    default_adapter_mode = os.getenv("DEFAULT_ADAPTER_MODE", "derma")

    cfg_path = os.path.join(default_adapter_path, "adapter_config.json")

    if os.path.isdir(default_adapter_path) and os.path.exists(cfg_path):
        log.info(f"[startup] Auto-loading adapter: {default_adapter_path} (mode={default_adapter_mode})")
        try:
            slm_component = await run_in_threadpool(
                lambda: DomainSLMComponent(
                    default_model,
                    VECTOR_DIR,
                    default_adapter_path,
                    default_adapter_mode,
                )
            )
            current_mode  = "finetune"
            current_model = default_model
            log.info(f"[startup] ✅ Adapter loaded successfully: {default_adapter_path}")
        except Exception:
            log.error(f"[startup] ❌ Failed to auto-load adapter:\n{traceback.format_exc()}")
    else:
        log.warning(
            f"[startup] ⚠️  No adapter found at '{default_adapter_path}' — "
            f"starting without a model. Call /node/load-adapter to load one."
        )

    yield  # ← app runs here


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="Chatbot SLM Component", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ── Schemas ───────────────────────────────────────────────────────────────────
class ModeRequest(BaseModel):
    model: str

class BioRagRequest(BaseModel):
    model: str
    chroma_path: str = "rag/chroma_db"
    lora_path: Optional[str] = None

class LoadAdapterRequest(BaseModel):
    model: str
    adapter_path: str
    mode: str = "generic"

class FineTuneRequest(BaseModel):
    model: str
    customization_type: str = Field(
        default="LoRA",
        description="One of: 'LoRA', 'RAG', 'Both'"
    )
    samples_per_chunk: int = Field(default=2, ge=1, le=10)
    lora_rank: int            = Field(default=8,    ge=2,   le=64)
    lora_alpha: int           = Field(default=16,   ge=1,   le=128)
    lora_dropout: float       = Field(default=0.05, ge=0.0, le=0.2)
    target_modules: List[str] = Field(
        default=["q_proj", "v_proj"],
        description="Which transformer layers get LoRA adapters"
    )
    learning_rate: float  = Field(default=2e-4, gt=0)
    batch_size: int       = Field(default=1,    ge=1, le=8)
    epochs: int           = Field(default=1,    ge=1, le=10)
    optimizer: str        = Field(default="adamw_torch")
    gradient_checkpointing: bool = Field(default=True)
    model_name: str            = Field(default="")
    model_version: str         = Field(default="v1.0")
    output_path: Optional[str] = Field(default=None)


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
        gc.collect()
        if HAS_GPU:
            torch.cuda.empty_cache()
        print("[api] Model unloaded. RAM freed.")
    else:
        gc.collect()
        if HAS_GPU:
            torch.cuda.empty_cache()


def _adapter_dir(req: FineTuneRequest) -> str:
    if req.output_path:
        path = req.output_path
        if os.path.normpath(path) == os.path.normpath(ADAPTERS_DIR):
            safe_name = re.sub(r"[^\w\-]", "_", req.model_name or req.model)
            safe_ver  = re.sub(r"[^\w\-]", "_", req.model_version or "v1")
            path = os.path.join(ADAPTERS_DIR, f"{safe_name}_{safe_ver}")
        return path

    safe_name = re.sub(r"[^\w\-]", "_", req.model_name or req.model)
    safe_ver  = re.sub(r"[^\w\-]", "_", req.model_version or "v1")
    folder = f"{safe_name}_{safe_ver}".strip("_") or f"{req.model}_adapter"
    return os.path.join(ADAPTERS_DIR, folder)


def _list_saved_adapters() -> list:
    adapters = []
    if not os.path.isdir(ADAPTERS_DIR):
        return adapters
    for entry in sorted(os.listdir(ADAPTERS_DIR)):
        full = os.path.join(ADAPTERS_DIR, entry)
        cfg  = os.path.join(full, "adapter_config.json")
        if os.path.isdir(full) and os.path.exists(cfg):
            try:
                with open(cfg) as f:
                    meta = json.load(f)
                adapters.append({
                    "name":           entry,
                    "path":           full,
                    "base_model":     meta.get("base_model_name_or_path", "unknown"),
                    "lora_rank":      meta.get("r", "?"),
                    "target_modules": meta.get("target_modules", []),
                })
            except Exception:
                adapters.append({"name": entry, "path": full})
    return adapters


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    import psutil
    mem = psutil.virtual_memory()
    return {
        "status":        "ok",
        "gpu_available": HAS_GPU,
        "ram_total_gb":  round(mem.total     / 1e9, 2),
        "ram_used_gb":   round(mem.used      / 1e9, 2),
        "ram_free_gb":   round(mem.available / 1e9, 2),
        "model_loaded":  current_model,
        "mode":          current_mode,
        "timestamp":     datetime.datetime.utcnow().isoformat(),
    }


@app.post("/node/bio-rag")
async def bio_rag_node(req: BioRagRequest):
    global slm_component, current_mode, current_model

    _validate_model(req.model)
    _unload_current_model()

    try:
        slm_component = await run_in_threadpool(
            lambda: BioSLMComponent(
                model_name=req.model,
                chroma_path=req.chroma_path,
                lora_path=req.lora_path,
            )
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load Bio-RAG: {traceback.format_exc()}")

    current_mode  = "bio-rag"
    current_model = req.model

    return {
        "node":        "bio-rag",
        "status":      "ready",
        "model":       req.model,
        "chroma_path": req.chroma_path,
    }


@app.get("/models")
async def list_models():
    return {
        "models":        MODEL_CATALOG,
        "gpu_available": HAS_GPU,
        "recommended":   [m["id"] for m in MODEL_CATALOG if m["cpu_safe"] or HAS_GPU],
    }


@app.get("/adapters")
async def list_adapters():
    adapters = _list_saved_adapters()
    return {
        "adapters": adapters,
        "count":    len(adapters),
    }


@app.post("/node/upload")
async def upload_document(files: List[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="No files provided.")

    saved_paths    = []
    uploaded_names = []
    skipped        = []

    for file in files:
        ext = os.path.splitext(file.filename or "")[1].lower()
        if ext not in (".pdf", ".docx"):
            skipped.append(file.filename)
            continue
        path = os.path.join(UPLOAD_DIR, file.filename)
        with open(path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        saved_paths.append(path)
        uploaded_names.append(file.filename)

    if not saved_paths:
        raise HTTPException(
            status_code=400,
            detail=f"No valid files to index. Only PDF and DOCX are supported. Skipped: {skipped}",
        )

    await run_in_threadpool(build_index, saved_paths, VECTOR_DIR)

    return {
        "node":    "upload",
        "status":  "indexed",
        "files":   uploaded_names,
        "count":   len(uploaded_names),
        "skipped": skipped,
    }


@app.post("/node/rag")
async def rag_node(req: ModeRequest):
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
    global slm_component, current_mode, current_model

    _validate_model(req.model)

    try:
        import psutil
        mem = psutil.virtual_memory()
        print(
            f"[api] RAM before finetune → "
            f"free={round(mem.available / 1e9, 2)} GB / "
            f"total={round(mem.total / 1e9, 2)} GB"
        )
    except Exception:
        pass

    if req.customization_type == "RAG":
        _unload_current_model()
        slm_component = await run_in_threadpool(
            lambda: SLMComponent(model_name=req.model, vector_dir=VECTOR_DIR, top_k=5)
        )
        current_mode  = "rag"
        current_model = req.model
        return {
            "node":   "finetune",
            "status": "ready",
            "mode":   "RAG only (no fine-tuning)",
            "model":  req.model,
        }

    _unload_current_model()

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

    docs = await run_in_threadpool(load_documents, file_paths)
    if not docs:
        raise HTTPException(
            status_code=400,
            detail="Could not extract text from uploaded documents.",
        )

    n_samples = await run_in_threadpool(
        build_domain_dataset, docs, DATASET_PATH, 80, req.samples_per_chunk, 0.15,
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

    lora_params = {
        "lora_rank":              req.lora_rank,
        "lora_alpha":             req.lora_alpha,
        "lora_dropout":           req.lora_dropout,
        "target_modules":         req.target_modules,
        "learning_rate":          req.learning_rate,
        "batch_size":             req.batch_size,
        "epochs":                 req.epochs,
        "optimizer":              req.optimizer,
        "gradient_checkpointing": req.gradient_checkpointing,
    }

    lora_dir = _adapter_dir(req)
    os.makedirs(lora_dir, exist_ok=True)
    print(f"[api] Adapter will be saved → {lora_dir}")

    try:
        await run_in_threadpool(train_domain_lora, req.model, dataset, lora_dir, lora_params)
    except MemoryError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Training failed:\n{traceback.format_exc()}",
        )

    try:
        slm_component = await run_in_threadpool(
            lambda: DomainSLMComponent(req.model, VECTOR_DIR, lora_dir)
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Fine-tune succeeded but failed to reload model:\n{traceback.format_exc()}",
        )

    current_mode  = "finetune"
    current_model = req.model

    return {
        "node":             "finetune",
        "status":           "ready",
        "model":            req.model,
        "customization":    req.customization_type,
        "training_samples": n_samples,
        "adapter_path":     lora_dir,
        "lora_rank":        req.lora_rank,
        "lora_alpha":       req.lora_alpha,
        "target_modules":   req.target_modules,
    }


@app.post("/node/load-adapter")
async def load_adapter_node(req: LoadAdapterRequest):
    """Load a previously saved LoRA adapter by path."""
    global slm_component, current_mode, current_model

    _validate_model(req.model)

    if not os.path.isdir(req.adapter_path):
        raise HTTPException(
            status_code=404,
            detail=f"Adapter not found at path: {req.adapter_path}",
        )
    cfg_path = os.path.join(req.adapter_path, "adapter_config.json")
    if not os.path.exists(cfg_path):
        raise HTTPException(
            status_code=400,
            detail="Path exists but is not a valid LoRA adapter (missing adapter_config.json).",
        )

    _unload_current_model()

    try:
        slm_component = await run_in_threadpool(
            lambda: DomainSLMComponent(req.model, VECTOR_DIR, req.adapter_path, req.mode)
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load adapter:\n{traceback.format_exc()}",
        )

    current_mode  = "finetune"
    current_model = req.model

    return {
        "node":         "load-adapter",
        "status":       "ready",
        "model":        req.model,
        "adapter_path": req.adapter_path,
        "mode":         req.mode,
    }


@app.post("/node/inference", response_model=QueryResponse)
async def inference_node(req: QueryRequest):
    global slm_component

    if slm_component is None:
        await asyncio.sleep(3)
        if slm_component is None:
            raise HTTPException(
                status_code=400,
                detail="No model loaded. Call /node/load-adapter or /node/rag first.",
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


@app.get("/node/uploadDocument")
async def upload_document_by_minio(file_path: str):
    log.debug(f"file uploaded {file_path}")
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = os.path.join(tmpdir, "file.pdf")
        try:
            MINIO_CLIENT.fget_object(NODE_STORAGE_REF, file_path, tmp_path)
            await run_in_threadpool(build_index, [tmp_path], VECTOR_DIR)
        except Exception:
            MINIO_CLIENT.fget_object(
                NODE_STORAGE_REF,
                f"test_input/{os.path.basename(tmp_path)}",
                tmp_path,
            )

    log.debug(f"file uploaded {file_path}")
    return {"node": "upload", "status": "uploaded", "file": file_path}
