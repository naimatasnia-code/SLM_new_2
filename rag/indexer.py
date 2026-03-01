

import os
import re

from langchain_community.document_loaders import PyPDFLoader, UnstructuredWordDocumentLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# ── shared embedding (imported by retriever too) ──────────────────────────────
EMBED_MODEL = "BAAI/bge-small-en-v1.5"

def _get_embeddings():
    return HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},   # cosine ready
    )


# ── text cleaning ─────────────────────────────────────────────────────────────
def _clean(text: str) -> str:
    text = re.sub(r"-\n", "", text)          # fix hyphenated line-breaks
    text = re.sub(r"\n+", " ", text)         # collapse newlines → space
    text = re.sub(r"\s{2,}", " ", text)      # collapse multiple spaces
    return text.strip()


# ── document loading ──────────────────────────────────────────────────────────
def load_documents(file_paths: list[str]):
    docs = []
    for path in file_paths:
        lower = path.lower()
        try:
            if lower.endswith(".pdf"):
                loader = PyPDFLoader(path)
            elif lower.endswith(".docx"):
                loader = UnstructuredWordDocumentLoader(path)
            else:
                print(f"[indexer] Skipping unsupported file: {path}")
                continue

            loaded = loader.load()
            # clean page_content in-place
            for doc in loaded:
                doc.page_content = _clean(doc.page_content)
            docs.extend(loaded)
            print(f"[indexer] Loaded {len(loaded)} pages from: {path}")
        except Exception as e:
            print(f"[indexer] ERROR loading {path}: {e}")
    return docs


# ── index builder ─────────────────────────────────────────────────────────────
def build_index(file_paths: list[str], vector_dir: str):
    all_docs = load_documents(file_paths)
    if not all_docs:
        raise ValueError("No text could be loaded from the provided documents.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,        # up from 300 → more context per chunk
        chunk_overlap=100,     # up from 50  → less info loss at boundaries
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(all_docs)
    if not chunks:
        raise ValueError("Text loaded but chunking produced no output.")

    print(f"[indexer] Total chunks: {len(chunks)}")
    embeddings = _get_embeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)

    os.makedirs(vector_dir, exist_ok=True)
    vectorstore.save_local(vector_dir)
    print(f"[indexer] Vector DB saved → {vector_dir}")
