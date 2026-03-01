

import time
from rag.retriever import load_retriever
from model.slm_loader import load_slm
from core.agent import MedicalAgent


class SLMComponent:
    def __init__(
        self,
        model_name: str,
        vector_dir: str | None = None,
        use_rag: bool = True,
        lora_path: str | None = None,
        top_k: int = 5,           # was 3 → 5 for better semantic recall
    ):
        self.tokenizer, self.model = load_slm(
            model_name,
            lora_path=lora_path,
            quantized=True,
        )

        retriever = None
        if use_rag and vector_dir:
            retriever = load_retriever(vector_dir, top_k)

        self.agent = MedicalAgent(self.tokenizer, self.model, retriever)

    def run(self, question: str) -> dict:
        start = time.time()
        out = self.agent.answer(question)
        out["latency_sec"] = round(time.time() - start, 3)
        out["model"] = self.model.config._name_or_path
        return out
