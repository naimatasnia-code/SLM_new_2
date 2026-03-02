# rag/bio_retriever.py

import os
from langchain_core.documents import Document
from .rag_retriever import BioRAG


class BioScoredRetriever:
    """
    Wraps BioRAG to match ScoredRetriever interface.
    """

    def __init__(self, chroma_path: str | None = None, n_results: int = 2):
        # Default to /app/rag/chroma_db (robust in Docker)
        if chroma_path is None:
            chroma_path = os.path.join(os.path.dirname(__file__), "chroma_db")

        self.n_results = n_results
        self.rag = BioRAG(db_path=chroma_path)

    def invoke(self, query: str) -> tuple[list, float]:
        results = self.rag.collection.query(
            query_texts=[query],
            n_results=self.n_results,
            include=["documents", "metadatas", "distances"],
        )

        if not results.get("documents") or not results["documents"][0]:
            return [], 0.0

        docs = []
        scores = []

        for i, text in enumerate(results["documents"][0]):
            meta = (results.get("metadatas") or [[]])[0][i] or {}
            distance = (results.get("distances") or [[]])[0][i]

            docs.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": meta.get("source", ""),
                        "type": meta.get("type", ""),
                        "topic": meta.get("topic", "N/A"),
                    },
                )
            )

            score = 1 / (1 + float(distance))
            scores.append(score)

        return docs, (max(scores) if scores else 0.0)
