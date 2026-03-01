
import time
from model.universal_loader import load_model
from rag.retriever import load_retriever
from core.agent import DocumentAgent


class DomainSLMComponent:
    def __init__(self, model_name: str, vector_dir: str, lora_path: str | None = None):
        self.tokenizer, self.model = load_model(model_name, lora_path)
        retriever = load_retriever(vector_dir, top_k=5)
        self.agent = DocumentAgent(self.tokenizer, self.model, retriever)

    def run(self, question: str) -> dict:
        start = time.time()
        out = self.agent.answer(question)
        out["latency_sec"] = round(time.time() - start, 3)
        out["model"] = self.model.config._name_or_path
        out["domain_adapter"] = True
        return out
