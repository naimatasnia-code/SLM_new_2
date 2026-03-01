
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from rag.indexer import EMBED_MODEL


def _get_embeddings():
    return HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def load_retriever(vector_dir: str, top_k: int = 5):
    """Returns a ScoredRetriever wrapper (not a plain LangChain retriever)."""
    embeddings = _get_embeddings()
    db = FAISS.load_local(
        vector_dir,
        embeddings,
        allow_dangerous_deserialization=True,
    )
    return ScoredRetriever(db, top_k)


class ScoredRetriever:
    """Thin wrapper that returns docs + relevance scores."""

    # Below this cosine-based score the context is considered too weak
    CONFIDENCE_THRESHOLD = 0.30

    def __init__(self, db: FAISS, top_k: int):
        self.db = db
        self.top_k = top_k

    def invoke(self, query: str) -> tuple[list, float]:
        """
        Returns:
            docs  – list of LangChain Document objects (best chunks)
            best_score – float, highest relevance score found
        """
        results = self.db.similarity_search_with_relevance_scores(
            query, k=self.top_k
        )
        if not results:
            return [], 0.0

        # sort descending by score, keep top_k
        results.sort(key=lambda x: x[1], reverse=True)
        docs = [r[0] for r in results]
        best_score = results[0][1]
        return docs, best_score
