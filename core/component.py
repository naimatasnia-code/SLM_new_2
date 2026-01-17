
import time
from rag.retriever import load_retriever
from model.slm_loader import load_slm
from core.agent import MedicalAgent

class SLMComponent:
   

    def __init__(self, model_name: str, vector_dir: str,top_k: int = 3):
        self.tokenizer, self.model = load_slm(model_name)
        self.retriever = load_retriever(vector_dir,top_k)
        self.agent = MedicalAgent(self.tokenizer, self.model, self.retriever)

    def run(self, question: str):
        start = time.time()
        out = self.agent.answer(question)

        return {
            "answer": out["answer"],
            "latency_sec": round(time.time() - start, 3),
            "prompt_tokens": out["prompt_tokens"],
            "completion_tokens": out["completion_tokens"],
            "total_tokens": out["total_tokens"],
            "model": self.model.config._name_or_path
        }
