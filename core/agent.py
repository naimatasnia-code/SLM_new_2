"""
core/agent.py  –  Updated
Changes vs original:
- Handles all query intents: greeting / thanks / farewell / rag / out-of-scope
- Confidence threshold: if best retrieval score < threshold → "not in documents"
- max_new_tokens raised to 400 (no more truncated answers)
- Response post-processing strips raw escape sequences and cleans whitespace
- Temperature 0.2 for factual, structured answers
- repetition_penalty added to stop looping
- Query expansion: appends synonyms to improve semantic recall
"""

import re
import torch

from core.prompt import build_prompt, build_chat_response
from rag.retriever import ScoredRetriever

# ── Synonym expansion map ─────────────────────────────────────────────────────
# Broadens queries so semantically similar terms get retrieved correctly
SYNONYM_MAP = {
    "recover": ["recover", "cure", "treat", "heal", "therapy", "treatment", "recovery"],
    "cure": ["cure", "treat", "recover", "therapy", "treatment", "healing"],
    "prevent": ["prevent", "avoid", "reduce risk", "protection", "preventive"],
    "symptom": ["symptom", "sign", "indication", "manifestation"],
    "cause": ["cause", "reason", "origin", "etiology", "factor"],
    "diagnosis": ["diagnosis", "detect", "identify", "test", "screening"],
    "medicine": ["medicine", "drug", "medication", "pharmaceutical"],
    "side effect": ["side effect", "adverse effect", "complication", "risk"],
}

# Intent keywords
_GREETINGS = {"hi", "hello", "hey", "good morning", "good afternoon", "good evening", "howdy"}
_THANKS     = {"thanks", "thank you", "thx", "ty", "appreciate it", "thank"}
_FAREWELL   = {"bye", "goodbye", "see you", "exit", "quit", "farewell"}


def _classify_intent(question: str) -> str:
    q = question.lower().strip().rstrip("!?.")
    if q in _GREETINGS or any(q.startswith(g) for g in _GREETINGS):
        return "greeting"
    if q in _THANKS:
        return "thanks"
    if q in _FAREWELL:
        return "farewell"
    return "rag"


def _expand_query(question: str) -> str:
    """Append synonyms of key terms to the query to improve retrieval recall."""
    q_lower = question.lower()
    extras = []
    for keyword, synonyms in SYNONYM_MAP.items():
        if keyword in q_lower:
            extras.extend(s for s in synonyms if s not in q_lower)
    if extras:
        return question + " " + " ".join(extras[:6])   # cap expansion length
    return question


def _clean_response(text: str) -> str:
    """Remove raw escape sequences and normalize whitespace."""
    text = text.replace("\\n", " ").replace("\\t", " ").replace("\\r", " ")
    text = re.sub(r"\n{3,}", "\n\n", text)       # max two consecutive newlines
    text = re.sub(r" {2,}", " ", text)            # collapse spaces
    text = re.sub(r"\s+([.,!?;:])", r"\1", text) # fix space-before-punctuation
    # Strip incomplete trailing sentence (ends without . ! ?)
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    if sentences and not re.search(r'[.!?]$', sentences[-1].strip()):
        sentences = sentences[:-1]
    return " ".join(sentences).strip() if sentences else text.strip()


# ── Agent ─────────────────────────────────────────────────────────────────────
class MedicalAgent:

    CONFIDENCE_THRESHOLD = 0.30   # min retrieval score to attempt an answer

    def __init__(self, tokenizer, model, retriever: ScoredRetriever | None):
        self.tokenizer = tokenizer
        self.model = model
        self.retriever = retriever

    # ── public entry point ────────────────────────────────────────────────────
    def answer(self, question: str) -> dict:
        intent = _classify_intent(question)

        # Handle non-RAG intents immediately (no LLM call needed → fast + cheap)
        if intent != "rag":
            return self._static_response(build_chat_response(intent), question)

        # No retriever loaded (shouldn't happen, but guard anyway)
        if self.retriever is None:
            return self._static_response(build_chat_response("no_docs"), question)

        # ── Retrieval ─────────────────────────────────────────────────────────
        expanded_query = _expand_query(question)
        docs, best_score = self.retriever.invoke(expanded_query)

        # Out-of-scope: context too weak
        if not docs or best_score < self.CONFIDENCE_THRESHOLD:
            return self._static_response(build_chat_response("out_of_scope"), question)

        # Build context from top chunks (keep within ~1200 chars to save memory)
        context_parts = []
        budget = 1200
        for doc in docs:
            chunk = doc.page_content.strip()
            if len(chunk) <= budget:
                context_parts.append(chunk)
                budget -= len(chunk)
            if budget <= 0:
                break
        context = "\n\n".join(context_parts)

        # ── Generation ────────────────────────────────────────────────────────
        prompt = build_prompt(context, question)
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1024,
        ).to(self.model.device)

        prompt_len = inputs["input_ids"].shape[-1]

        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=400,          # was 200 → fixes truncation
                do_sample=False,             # greedy = deterministic, factual
                temperature=1.0,             # ignored when do_sample=False
                repetition_penalty=1.15,     # stops the model looping
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        generated_ids = output[0][prompt_len:]
        raw_answer = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        clean_answer = _clean_response(raw_answer)

        # If model echoed the prompt or produced nothing, fall back
        if not clean_answer or len(clean_answer) < 10:
            clean_answer = build_chat_response("out_of_scope")

        return {
            "answer": clean_answer,
            "prompt_tokens": prompt_len,
            "completion_tokens": len(generated_ids),
            "total_tokens": prompt_len + len(generated_ids),
        }

    # ── helpers ───────────────────────────────────────────────────────────────
    @staticmethod
    def _static_response(text: str, question: str) -> dict:
        return {
            "answer": text,
            "prompt_tokens": len(question.split()),
            "completion_tokens": len(text.split()),
            "total_tokens": len(question.split()) + len(text.split()),
        }
