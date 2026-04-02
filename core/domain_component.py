import os
import time
from model.universal_loader import load_model
from rag.retriever import load_retriever
from core.agent import DocumentAgent

class DomainSLMComponent:
    def __init__(self, model_name: str, vector_dir: str,
                 lora_path: str | None = None, mode: str = "generic"):  # ← add mode
        self.tokenizer, self.model = load_model(model_name, lora_path)

        faiss_index = os.path.join(vector_dir, "index.faiss")
        if os.path.exists(faiss_index):
            retriever    = load_retriever(vector_dir, top_k=5)
            adapter_only = False
        else:
            retriever    = None
            adapter_only = True

        self.agent = DocumentAgent(
            self.tokenizer, self.model, retriever, adapter_only, mode  #  pass mode
        )

    def run(self, question: str) -> dict:
        start = time.time()
        out = self.agent.answer(question)
        out["latency_sec"]    = round(time.time() - start, 3)
        out["model"]          = self.model.config._name_or_path
        out["domain_adapter"] = True
        return out