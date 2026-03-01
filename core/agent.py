

import re
import torch

from core.prompt import build_prompt, build_chat_response
from rag.retriever import ScoredRetriever

# ── Synonym expansion map ─────────────────────────────────────────────────────
SYNONYM_MAP = {
    # Medical
    "recover":     ["recover", "cure", "treat", "heal", "therapy", "treatment", "recovery"],
    "cure":        ["cure", "treat", "recover", "therapy", "treatment", "healing"],
    "prevent":     ["prevent", "avoid", "reduce risk", "protection", "preventive"],
    "symptom":     ["symptom", "sign", "indication", "manifestation"],
    "cause":       ["cause", "reason", "origin", "etiology", "factor"],
    "diagnosis":   ["diagnosis", "detect", "identify", "test", "screening"],
    "medicine":    ["medicine", "drug", "medication", "pharmaceutical"],
    "side effect": ["side effect", "adverse effect", "complication", "risk"],
    # Financial
    "cost":        ["cost", "price", "fee", "charge", "expense", "rate", "amount"],
    "profit":      ["profit", "gain", "revenue", "income", "earnings", "return"],
    "loss":        ["loss", "deficit", "shortfall", "negative return"],
    "budget":      ["budget", "allocation", "forecast", "estimate", "plan"],
    "pay":         ["pay", "salary", "compensation", "wage", "remuneration"],
    # Legal
    "rule":        ["rule", "regulation", "policy", "law", "requirement", "clause"],
    "agreement":   ["agreement", "contract", "deal", "terms", "arrangement"],
    "right":       ["right", "entitlement", "privilege", "permission", "authority"],
    "obligation":  ["obligation", "duty", "responsibility", "requirement"],
    "penalty":     ["penalty", "fine", "sanction", "consequence", "punishment"],
    # HR
    "employee":    ["employee", "staff", "worker", "personnel", "team member"],
    "hire":        ["hire", "recruit", "onboard", "employ", "appoint"],
    "fire":        ["fire", "terminate", "dismiss", "let go", "redundancy"],
    "leave":       ["leave", "absence", "vacation", "time off", "holiday"],
    "performance": ["performance", "appraisal", "review", "evaluation", "assessment"],
    # Technical
    "error":       ["error", "bug", "issue", "fault", "failure", "problem", "exception"],
    "fix":         ["fix", "resolve", "repair", "patch", "solution", "troubleshoot"],
    "install":     ["install", "setup", "configure", "deploy", "set up"],
    "update":      ["update", "upgrade", "patch", "version", "release"],
    "feature":     ["feature", "functionality", "capability", "option", "module"],
    # General
    "explain":     ["explain", "describe", "define", "what is", "overview"],
    "compare":     ["compare", "difference", "vs", "versus", "contrast"],
    "summarize":   ["summarize", "summary", "overview", "brief", "key points"],
    "steps":       ["steps", "process", "procedure", "how to", "instructions"],
    "benefits":    ["benefits", "advantages", "pros", "positive", "gains"],
    "drawbacks":   ["drawbacks", "disadvantages", "cons", "negative", "limitations"],
}

_GREETINGS = {"hi", "hello", "hey", "good morning", "good afternoon", "good evening", "howdy"}
_THANKS    = {"thanks", "thank you", "thx", "ty", "appreciate it", "thank"}
_FAREWELL  = {"bye", "goodbye", "see you", "exit", "quit", "farewell"}


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
    q_lower = question.lower()
    extras = []
    for keyword, synonyms in SYNONYM_MAP.items():
        if keyword in q_lower:
            extras.extend(s for s in synonyms if s not in q_lower)
    return (question + " " + " ".join(extras[:8])).strip() if extras else question


def _clean_response(text: str) -> str:
    text = text.replace("\\n", " ").replace("\\t", " ").replace("\\r", " ")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" {2,}", " ", text)
    text = re.sub(r"\s+([.,!?;:])", r"\1", text)
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    if sentences and not re.search(r'[.!?]$', sentences[-1].strip()):
        sentences = sentences[:-1]
    return " ".join(sentences).strip() if sentences else text.strip()


class DocumentAgent:
    """Domain-agnostic RAG agent — works with any uploaded document."""

    CONFIDENCE_THRESHOLD = 0.30

    def __init__(self, tokenizer, model, retriever: ScoredRetriever | None):
        self.tokenizer = tokenizer
        self.model     = model
        self.retriever = retriever

    def answer(self, question: str) -> dict:
        intent = _classify_intent(question)

        if intent != "rag":
            return self._static_response(build_chat_response(intent), question)

        if self.retriever is None:
            return self._static_response(build_chat_response("no_docs"), question)

        expanded_query = _expand_query(question)
        docs, best_score = self.retriever.invoke(expanded_query)

        if not docs or best_score < self.CONFIDENCE_THRESHOLD:
            return self._static_response(build_chat_response("out_of_scope"), question)

        context_parts, budget = [], 1200
        for doc in docs:
            chunk = doc.page_content.strip()
            if len(chunk) <= budget:
                context_parts.append(chunk)
                budget -= len(chunk)
            if budget <= 0:
                break
        context = "\n\n".join(context_parts)

        prompt = build_prompt(context, question)
        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=1024,
        ).to(self.model.device)
        prompt_len = inputs["input_ids"].shape[-1]

        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=400,
                do_sample=False,
                repetition_penalty=1.15,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        generated_ids = output[0][prompt_len:]
        raw_answer    = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        clean_answer  = _clean_response(raw_answer)

        if not clean_answer or len(clean_answer) < 10:
            clean_answer = build_chat_response("out_of_scope")

        return {
            "answer":            clean_answer,
            "prompt_tokens":     prompt_len,
            "completion_tokens": len(generated_ids),
            "total_tokens":      prompt_len + len(generated_ids),
        }

    @staticmethod
    def _static_response(text: str, question: str) -> dict:
        return {
            "answer":            text,
            "prompt_tokens":     len(question.split()),
            "completion_tokens": len(text.split()),
            "total_tokens":      len(question.split()) + len(text.split()),
        }


# Backwards-compatible alias — any existing code importing MedicalAgent still works
MedicalAgent = DocumentAgent
