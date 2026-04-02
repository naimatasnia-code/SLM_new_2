import re
import torch

from core.prompt import build_prompt, build_chat_response
from rag.retriever import ScoredRetriever

# ── Synonym map ───────────────────────────────────────────────────────────────
SYNONYM_MAP = {
    "recover":     ["recover", "cure", "treat", "heal", "therapy", "treatment", "recovery"],
    "cure":        ["cure", "treat", "recover", "therapy", "treatment", "healing"],
    "prevent":     ["prevent", "avoid", "reduce risk", "protection", "preventive"],
    "symptom":     ["symptom", "sign", "indication", "manifestation"],
    "cause":       ["cause", "reason", "origin", "etiology", "factor"],
    "diagnosis":   ["diagnosis", "detect", "identify", "test", "screening"],
    "medicine":    ["medicine", "drug", "medication", "pharmaceutical"],
    "side effect": ["side effect", "adverse effect", "complication", "risk"],
    "cost":        ["cost", "price", "fee", "charge", "expense", "rate", "amount"],
    "profit":      ["profit", "gain", "revenue", "income", "earnings", "return"],
    "loss":        ["loss", "deficit", "shortfall", "negative return"],
    "budget":      ["budget", "allocation", "forecast", "estimate", "plan"],
    "pay":         ["pay", "salary", "compensation", "wage", "remuneration"],
    "rule":        ["rule", "regulation", "policy", "law", "requirement", "clause"],
    "agreement":   ["agreement", "contract", "deal", "terms", "arrangement"],
    "right":       ["right", "entitlement", "privilege", "permission", "authority"],
    "obligation":  ["obligation", "duty", "responsibility", "requirement"],
    "penalty":     ["penalty", "fine", "sanction", "consequence", "punishment"],
    "employee":    ["employee", "staff", "worker", "personnel", "team member"],
    "hire":        ["hire", "recruit", "onboard", "employ", "appoint"],
    "fire":        ["fire", "terminate", "dismiss", "let go", "redundancy"],
    "leave":       ["leave", "absence", "vacation", "time off", "holiday"],
    "performance": ["performance", "appraisal", "review", "evaluation", "assessment"],
    "error":       ["error", "bug", "issue", "fault", "failure", "problem", "exception"],
    "fix":         ["fix", "resolve", "repair", "patch", "solution", "troubleshoot"],
    "install":     ["install", "setup", "configure", "deploy", "set up"],
    "update":      ["update", "upgrade", "patch", "version", "release"],
    "feature":     ["feature", "functionality", "capability", "option", "module"],
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

_STOP_MARKERS = [
    r"\[END_OF_TEXT\]", r"\[END\]", r"\[\/INST\]",
    r"Step-by-step reasoning\s*:",
    r"Note\s*:", r"Please note", r"Always consult", r"Disclaimer\s*:",
    r"This approach aligns with", r"Continuous assessment ensures",
    r"specific recommendations can vary",
    r"consult with a (doctor|physician|nephrologist|specialist|professional)",
    r"The above (information|answer|response) is (not|for)",
    r"I hope this helps", r"I hope that helps",
    r"Feel free to ask", r"Let me know if",
    r"Thank you for providing",
    r"Please go ahead and ask",
    r"Are you looking for",
    r"What do you want to know",
]


# ─────────────────────────────────────────────────────────────────────────────
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


def _context_is_relevant(query: str, context: str, threshold: float = 0.25) -> bool:
    STOPWORDS = {
        "what", "is", "are", "the", "a", "an", "of", "in", "to", "and",
        "or", "how", "why", "when", "where", "who", "does", "do", "can",
        "for", "on", "with", "that", "this", "it", "be", "was", "were",
        "has", "have", "had", "will", "would", "could", "should", "about",
        "tell", "me", "explain", "describe", "give", "list",
    }
    query_words = {
        w for w in re.findall(r'\b\w+\b', query.lower())
        if w not in STOPWORDS and len(w) > 2
    }
    if not query_words:
        return True

    context_lower = context.lower()
    matched = sum(1 for w in query_words if w in context_lower)
    overlap_ratio = matched / len(query_words)
    return overlap_ratio >= threshold


def _is_hallucinated(answer: str, context: str) -> bool:
    answer_lower = answer.lower()

    no_info_phrase = "i don't have information about this"
    if no_info_phrase in answer_lower:
        first_sentence_end = re.search(r'[.!?]', answer)
        if first_sentence_end:
            return True

    if len(answer.split()) > 40:
        STOPWORDS = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been",
            "have", "has", "had", "do", "does", "did", "will", "would",
            "could", "should", "may", "might", "shall", "can", "to", "of",
            "in", "on", "at", "by", "for", "with", "about", "as", "into",
            "through", "from", "and", "or", "but", "not", "this", "that",
            "it", "its", "they", "them", "their", "which", "who", "what",
        }
        context_words = {
            w for w in re.findall(r'\b\w+\b', context.lower())
            if w not in STOPWORDS and len(w) > 3
        }
        answer_words = {
            w for w in re.findall(r'\b\w+\b', answer_lower)
            if w not in STOPWORDS and len(w) > 3
        }
        if not context_words:
            return False
        overlap = len(answer_words & context_words) / max(len(answer_words), 1)
        if overlap < 0.15:
            return True

    return False


def _clean_response(text: str) -> str:
    for marker in _STOP_MARKERS:
        parts = re.split(marker, text, maxsplit=1, flags=re.IGNORECASE)
        if len(parts) > 1:
            text = parts[0]

    text = text.replace("\\n", "\n").replace("\\t", " ").replace("\\r", " ")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"[ \t]+([.,!?;:])", r"\1", text)

    paragraphs = text.strip().split("\n\n")
    cleaned = []
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        sentences = re.split(r'(?<=[.!?])\s+', para)
        if sentences and not re.search(r'[.!?]$', sentences[-1].strip()):
            sentences = sentences[:-1]
        if sentences:
            cleaned.append(" ".join(s.strip() for s in sentences))

    return "\n\n".join(cleaned).strip()


def _generate_from_model(tokenizer, model, question: str) -> dict:
    """Shared generation logic for adapter-only mode."""
    prompt = build_prompt("", question)
    inputs = tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=1024,
    ).to(model.device)
    prompt_len = inputs["input_ids"].shape[-1]

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=300,
            do_sample=False,
            repetition_penalty=1.2,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated_ids = output[0][prompt_len:]
    raw_answer    = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    clean_answer  = _clean_response(raw_answer)

    return {
        "answer":            clean_answer,
        "prompt_tokens":     prompt_len,
        "completion_tokens": len(generated_ids),
        "total_tokens":      prompt_len + len(generated_ids),
    }


# ─────────────────────────────────────────────────────────────────────────────
class DocumentAgent:
    """
    Domain-agnostic agent.
    - adapter_only=True  → generates directly from adapter, no RAG needed
    - adapter_only=False → RAG retrieval + generation (existing behaviour)
    """

    CONFIDENCE_THRESHOLD = 0.45

    def __init__(self, tokenizer, model, retriever: "ScoredRetriever | None", adapter_only: bool = False):
        self.tokenizer    = tokenizer
        self.model        = model
        self.retriever    = retriever
        self.adapter_only = adapter_only

    def answer(self, question: str) -> dict:
        intent = _classify_intent(question)
        if intent != "rag":
            return self._static_response(build_chat_response(intent), question)

        # ── Adapter-only mode ─────────────────────────────────────────────────
        if self.adapter_only or self.retriever is None:
            return _generate_from_model(self.tokenizer, self.model, question)

        # ── RAG mode (existing code unchanged) ────────────────────────────────
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

        if not _context_is_relevant(question, context):
            return self._static_response(build_chat_response("out_of_scope"), question)

        prompt = build_prompt(context, question)
        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=1024,
        ).to(self.model.device)
        prompt_len = inputs["input_ids"].shape[-1]

        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=300,
                do_sample=False,
                repetition_penalty=1.2,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        generated_ids = output[0][prompt_len:]
        raw_answer    = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        clean_answer  = _clean_response(raw_answer)

        if not clean_answer or len(clean_answer) < 10 or _is_hallucinated(clean_answer, context):
            return self._static_response(build_chat_response("out_of_scope"), question)

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


MedicalAgent = DocumentAgent