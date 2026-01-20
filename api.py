from fastapi import FastAPI
from pydantic import BaseModel
from core.component import SLMComponent

app = FastAPI(title="Medical Chatbot SLM Component")

# Initialize the SLM once at startup
slm = SLMComponent(
    model_name="phi-2",  # or "mistral"
    vector_dir="./vector_db"
)

class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str
    latency_sec: float
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    model: str

@app.post("/chat", response_model=QueryResponse)
def chat(req: QueryRequest):
    return slm.run(req.question)
