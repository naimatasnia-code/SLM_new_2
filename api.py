from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import os
import shutil

from rag.indexer import build_index
from core.component import SLMComponent

UPLOAD_DIR = "uploads"
VECTOR_DIR = "vector_db"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(VECTOR_DIR, exist_ok=True)

app = FastAPI(title="Medical Chatbot SLM Component")

slm_component = None


# ---------- Schemas ----------

class UploadResponse(BaseModel):
    status: str
    filename: str


class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    answer: str
    latency_sec: float
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    model: str


# ---------- Endpoints ----------

@app.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Build / overwrite index
    build_index(
        file_paths=[file_path],
        vector_dir=VECTOR_DIR
    )

    global slm_component
    slm_component = SLMComponent(
        model_name="phi-2",
        vector_dir=VECTOR_DIR
    )

    return {
        "status": "indexed",
        "filename": file.filename
    }


@app.post("/chat", response_model=QueryResponse)
def chat(req: QueryRequest):
    if slm_component is None:
        raise HTTPException(
            status_code=400,
            detail="No document indexed yet. Upload a document first."
        )

    return slm_component.run(req.question)
