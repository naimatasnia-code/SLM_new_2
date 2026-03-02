# rag/bio_retriever.py

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag_retriever import BioRAG
from langchain.schema import Document


class BioScoredRetriever:
    """
    Wraps BioRAG to match ScoredRetriever interface.
    DocumentAgent works with this unchanged.
    """

    def __init__(self, chroma_path: str = "./chroma_db_final"):
        self.rag = BioRAG(db_path=chroma_path)

    def invoke(self, query: str) -> tuple[list, float]:
        
        # Call ChromaDB directly to also get distances (similarity scores)
        results = self.rag.collection.query(
            query_texts=[query],
            n_results=2,
            include=["documents", "metadatas", "distances"]
        )

        if not results["documents"] or not results["documents"][0]:
            return [], 0.0

        docs = []
        scores = []

        for i in range(len(results["documents"][0])):
            doc = Document(
                page_content=results["documents"][0][i],
                metadata={
                    "source": results["metadatas"][0][i].get("source", ""),
                    "type":   results["metadatas"][0][i].get("type", ""),
                    "topic":  results["metadatas"][0][i].get("topic", "N/A"),
                }
            )
            docs.append(doc)

            # ChromaDB returns L2 distances (lower = more similar)
            # Convert to a 0-1 similarity score so DocumentAgent's
            # confidence threshold check works correctly
            distance = results["distances"][0][i]
            score = 1 / (1 + distance)
            scores.append(score)

        best_score = max(scores) if scores else 0.0
        return docs, best_score
