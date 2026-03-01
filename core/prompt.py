"""
core/prompt.py  –  Strict grounding prompt
"""


def build_prompt(context: str, question: str) -> str:
    return f"""You are a document assistant. You have NO knowledge of your own.
You can ONLY answer using the CONTEXT provided below. If the answer is not in the CONTEXT, say "I don't have information about this in the provided documents." and STOP.

STRICT RULES:
1. Use ONLY the CONTEXT. Never use outside knowledge.
2. Match meaning, not just exact words ("recover" = "cure" = "treat").
3. Answer in complete sentences. Use numbered points for lists.
4. Do NOT write \\n literally. Do NOT add disclaimers, notes, or reasoning steps.
5. Stop immediately after the answer is complete.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER (from CONTEXT only, stop when done):"""


def build_chat_response(intent: str, question: str = "") -> str:
    responses = {
        "greeting":     "Hello! I'm your document assistant. Upload any file and ask me anything about it.",
        "thanks":       "You're welcome! Feel free to ask more questions about your documents.",
        "farewell":     "Goodbye! Come back anytime you need help with your documents.",
        "no_docs":      "No documents have been uploaded yet. Please upload a file first.",
        "out_of_scope": "I don't have information about this in the provided documents.",
    }
    return responses.get(intent, "I'm here to help. Please ask a question about your uploaded documents.")
