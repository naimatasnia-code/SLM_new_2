from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import time, os, shutil, json

from rag.indexer import build_index, load_documents

from minio import Minio

import os
import shutil
import time
import asyncio
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware

import datetime

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
MINIO_API_HOST = os.getenv("MINIO_API_HOST", "192.168.1.10:9000")
ACCESS_KEY = os.getenv("ACCESS_KEY", "minio")
SECRET_KEY = os.getenv("SECRET_KEY", "password")
NODE_STORAGE_REF = os.getenv("NODE_STORAGE_REF", "nodes_bucket")

MINIO_CLIENT = Minio(
    MINIO_API_HOST,
    access_key=ACCESS_KEY,
    secret_key=SECRET_KEY,
    secure=False,
)

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(VECTOR_DIR, exist_ok=True)

app = FastAPI(title="Chatbot SLM Component")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

slm_component = None
current_mode = None
current_model = None

# Schemas
class ModeRequest(BaseModel):
    model: str  # "phi-2" | "tinyllama" | "qwen-0.5b"

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


# Upload Node
@app.post("/node/upload")
async def upload_document(file: UploadFile = File(...)):
    path = os.path.join(UPLOAD_DIR, file.filename)
    with open(path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Index inside upload (as requested)
    await run_in_threadpool(build_index, [path], VECTOR_DIR)

    return {"node": "upload", "status": "uploaded", "file": file.filename}


# 
# RAG Node
# 
@app.post("/node/rag")
async def rag_node(req: ModeRequest):
    global slm_component, current_mode, current_model

    slm_component = SLMComponent(model_name=req.model, vector_dir=VECTOR_DIR)
    current_mode = "rag"
    current_model = req.model

    return {"node": "rag", "status": "ready", "model": req.model}


# Fine-tune Node
@app.post("/node/finetune")
async def finetune_node(req: ModeRequest):
    global slm_component, current_mode, current_model

    # Load docs
    file_paths = [os.path.join(UPLOAD_DIR, f) for f in os.listdir(UPLOAD_DIR)]
    docs = await run_in_threadpool(load_documents, file_paths)

    # Build dataset
    await run_in_threadpool(build_domain_dataset, docs, DATASET_PATH)

    # Load dataset JSONL
    with open(DATASET_PATH) as f:
        data = [json.loads(l) for l in f]
    if len(data) == 0:
        raise HTTPException(500, "Dataset is empty. No training samples generated.")

    dataset = Dataset.from_list(data)

    # Train LoRA
    await run_in_threadpool(train_domain_lora, req.model, dataset, LORA_DIR)

    # Load adapted model
    slm_component = DomainSLMComponent(req.model, VECTOR_DIR, LORA_DIR)

    current_mode = "finetune"
    current_model = req.model

    return {"node": "finetune", "status": "ready", "model": req.model}


# Inference Node (raw generation)
@app.post("/node/inference", response_model=QueryResponse)
async def inference_node(req: QueryRequest):
    global slm_component

    if slm_component is None:
        print(F"Chat slm_component is None {datetime.datetime.now()}")
        await asyncio.sleep(5)
        print(F"Chat slm_component is None {datetime.datetime.now()}")
        raise HTTPException(
            status_code=400,
            detail="No document indexed yet. Upload a document first."
        )

    start = time.time()
    result = await run_in_threadpool(slm_component.run, req.question)
    result["latency_sec"] = round(time.time() - start, 3)
    result["domain_adapter"] = (current_mode == "finetune")

    return result


# Chat Node (alias of inference for UI)
@app.post("/chat", response_model=QueryResponse)
async def chat(req: QueryRequest):
    # Reuse inference node logic
    return await inference_node(req)
