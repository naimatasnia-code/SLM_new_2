"""
core/prompt.py
==============
Prompt templates and static chat responses.
"""


def build_prompt(context: str, question: str) -> str:
    return f"""You are an expert document assistant with strong reasoning ability.
You answer ONLY from the CONTEXT below. You have NO outside knowledge.
If the answer is not in the CONTEXT, say "I don't have information about this in the provided documents." and STOP.

STRICT RULES:
1. Use ONLY the CONTEXT. Never use outside knowledge.
2. Match meaning across synonyms ("recover" = "cure" = "treat", "biological age" = "epigenetic age").
3. If multiple CONTEXT sections are relevant, COMBINE them into one complete answer.
4. Reason step by step internally before writing your answer, but do NOT show reasoning steps.
5. Answer in clear, complete sentences. Use numbered points for multi-part answers.
6. Simulate how a real person would ask this — answer both the literal question AND its intent.
   Example: "Am I healthy?" means answer what the documents say about health markers, risks, and recommendations.
7. Do NOT write disclaimers, notes, "please consult", or reasoning steps in your answer.
8. Do NOT repeat the question back. Start directly with the answer.
9. Stop immediately when the answer is complete.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER (combine all relevant context, stop when complete):"""


def build_chat_response(intent: str, question: str = "") -> str:
    responses = {
        "greeting": (
            "Hi! I'm your document assistant. "
            "Upload a file and ask me anything about it — I'm here to help!"
        ),
        "thanks": (
            "You're welcome! Feel free to ask more questions about your documents."
        ),
        "farewell": (
            "Goodbye! Come back anytime you need help with your documents."
        ),
        "no_docs": (
            "Hi there! It looks like no documents have been uploaded yet. "
            "Please upload a PDF or DOCX file first and I'll be ready to answer your questions."
        ),
        "out_of_scope": (
            "I don't have information about this in the provided documents."
        ),
    }
    return responses.get(
        intent,
        "Hi! I'm here to help. Please tell me, how can I help you?."
    )