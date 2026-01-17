import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import UnstructuredWordDocumentLoader

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings


def load_documents(file_paths):
    docs = []

    for path in file_paths:
        path_lower = path.lower()

        if path_lower.endswith(".pdf"):
            print(f"Loading PDF: {path}")
            loader = PyPDFLoader(path)
            docs.extend(loader.load())

        elif path_lower.endswith(".docx"):
            print(f"Loading DOCX: {path}")
            loader = UnstructuredWordDocumentLoader(path)
            docs.extend(loader.load())

        else:
            print(f"Skipping unsupported file: {path}")

    return docs


def build_index(file_paths, vector_dir):
    all_docs = load_documents(file_paths)

    if not all_docs:
        raise ValueError("No text could be loaded from documents.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    chunks = splitter.split_documents(all_docs)

    if not chunks:
        raise ValueError("Text loaded but no chunks were created.")

    print(f"Total chunks created: {len(chunks)}")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_documents(chunks, embeddings)

    os.makedirs(vector_dir, exist_ok=True)
    vectorstore.save_local(vector_dir)

    print("Vector DB created at:", vector_dir)




