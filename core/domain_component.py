from model.universal_loader import load_model
from rag.retriever import load_retriever
from core.agent import MedicalAgent
import time

class DomainSLMComponent:
    def __init__(self, model_name, vector_dir, lora_path=None):
        self.tokenizer, self.model = load_model(model_name, lora_path)
        self.retriever = load_retriever(vector_dir, 3)
        self.agent = MedicalAgent(self.tokenizer, self.model, self.retriever)

    def run(self, question):
        start = time.time()
        out = self.agent.answer(question)
        return {
            "answer": out["answer"],
            "latency_sec": round(time.time() - start, 3),
            "prompt_tokens": out["prompt_tokens"],
            "completion_tokens": out["completion_tokens"],
            "total_tokens": out["total_tokens"],
            "model": self.model.config._name_or_path,
            "domain_adapter": True
        }

