import chromadb
from chromadb.utils import embedding_functions
import os

class BioRAG:
    def __init__(self, db_path=None):
        base_dir = os.path.dirname(__file__)   # /app/rag
        if db_path is None:
            db_path = os.path.join(base_dir, "chroma_db")  # /app/rag/chroma_db

        if not os.path.exists(db_path):
            raise FileNotFoundError(f"Database folder '{db_path}' not found.")

        self.client = chromadb.PersistentClient(path=db_path)
        self.embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        self.collection = self.client.get_collection(
            name="bio_rag_memory",
            embedding_function=self.embedding_func
        )

    def search(self, query, n_results=2):
        """
        it retrieves relevant context for the SLM Agent.
        Input: User query string.
        Output: List of dictionaries containing context and metadata.
        """
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results
        )
        
        formatted_results = []
        if results['documents'] and results['documents'][0]:
            for i in range(len(results['documents'][0])):
                formatted_results.append({
                    "context": results['documents'][0][i],
                    "source": results['metadatas'][0][i].get('source'),
                    "type": results['metadatas'][0][i].get('type'),
                    "topic": results['metadatas'][0][i].get('topic', 'N/A')
                })
        return formatted_results

# DEMO TEST

if __name__ == "__main__":
    try:
        rag = BioRAG()
        print("BIO-RAG SYSTEM CHECK (DEMO QUESTIONS)")

        # Question 1: Aging Gap (Dynamic Data)
        print("\nQ1: What is my biological age compared to my actual age?")
        res1 = rag.search("What is my biological age compared to my actual age?")
        print(f"Found in {res1[0]['source']}: {res1[0]['context']}")

        # Question 2: Lifestyle (Dynamic Data)
        print("\nQ2: Do I have a smoking signature in my DNA?")
        res2 = rag.search("Do I have a smoking signature in my DNA?")
        print(f"Found in {res2[0]['source']}: {res2[0]['context']}")

        # Question 3: Genetics (Static Data)
        print("\nQ3: Am I a sprinter or an endurance athlete?")
        res3 = rag.search("Am I a sprinter or an endurance athlete?")
        print(f"Found in {res3[0]['source']}: {res3[0]['context']}")

        # Question 4: The Combined Insight (Static + Dynamic)
        print("\nQ4: How are my longevity markers looking?")
        res4 = rag.search("How are my longevity markers looking?")
   
        for r in res4:
            print(f" Found in {r['source']}: {r['context']}")


    except Exception as e:
        print(f"ERROR: {e}")
