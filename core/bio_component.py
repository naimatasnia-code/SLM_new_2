# core/bio_component.py

import time
from model.slm_loader import load_slm
from rag.bio_retriever import BioScoredRetriever
from core.agent import DocumentAgent


class BioSLMComponent:
    

    def __init__(
        self,
        model_name: str,
        chroma_path: str = "rag/chroma_db",
        lora_path: str | None = None,
    ):
        self.tokenizer, self.model = load_slm(
            model_name,
            lora_path=lora_path,
            quantized=True,
        )
        retriever = BioScoredRetriever(chroma_path)
        self.agent = DocumentAgent(self.tokenizer, self.model, retriever)

    def run(self, question: str) -> dict:
        start = time.time()
        out   = self.agent.answer(question)
        out["latency_sec"] = round(time.time() - start, 3)
        out["model"]       = self.model.config._name_or_path
        out["domain"]      = "bio-rag"
        return out
