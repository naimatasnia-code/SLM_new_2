


def build_prompt(context: str, question: str) -> str:
    return f"""You are a precise document assistant. Answer questions using ONLY the context below.

RULES:
1. Read the context carefully. Words with similar meaning (e.g. "cure"/"recover", "cost"/"price", "staff"/"employees") should be understood as equivalent — match meaning, not just exact words.
2. If the context contains relevant information, answer clearly and completely in full sentences. Never cut off mid-sentence.
3. Structure your answer: use numbered steps or short paragraphs when listing multiple points.
4. Do NOT use raw escape characters like \\n in your output.
5. If the context does NOT contain enough information to answer the question, respond with exactly: "I don't have information about this in the provided documents."
6. Do not add assumptions, warnings, or information beyond what the document says.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:"""


def build_chat_response(intent: str, question: str = "") -> str:
    """Pre-built responses for non-RAG intents."""
    responses = {
        "greeting":     "Hello! I'm your document assistant. Upload any file and ask me anything about it.",
        "thanks":       "You're welcome! Feel free to ask more questions about your documents.",
        "farewell":     "Goodbye! Come back anytime you need help with your documents.",
        "no_docs":      "No documents have been uploaded yet. Please upload a file first so I can answer your questions.",
        "out_of_scope": "I don't have information about this in the provided documents.",
    }
    return responses.get(intent, "I'm here to help. Please ask a question about your uploaded documents.")
