

import json
import os
import re
import random
from typing import List

# ── Question templates per content type ──────────────────────────────────────
_FACTUAL_TEMPLATES = [
    "What does the document say about {topic}?",
    "Explain {topic} based on the provided document.",
    "Summarize what is mentioned about {topic}.",
    "What information is given about {topic}?",
    "Describe {topic} as mentioned in the document.",
    "According to the document, what is {topic}?",
    "What can you tell me about {topic} from this document?",
]

_DEFINITION_TEMPLATES = [
    "What is {topic}?",
    "Define {topic} as per the document.",
    "How is {topic} described in the document?",
]

_NEGATIVE_TOPICS = [
    "quantum computing", "stock market", "weather forecast",
    "recipe for pasta", "Java programming language",
    "football scores", "cryptocurrency prices",
    "celebrity gossip", "ancient history",
]

_NEGATIVE_RESPONSE = "I don't have information about this in the provided documents."


def _extract_topic(text: str, max_words: int = 5) -> str:
    """
    Pulls the most meaningful noun phrase from a text chunk.
    Falls back to first N non-stopword words.
    """
    STOPWORDS = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been",
        "have", "has", "had", "do", "does", "did", "will", "would",
        "could", "should", "may", "might", "shall", "can", "to", "of",
        "in", "on", "at", "by", "for", "with", "about", "as", "into",
        "through", "from", "and", "or", "but", "not", "this", "that",
        "it", "its", "they", "them", "their", "which", "who", "what",
        "all", "each", "both", "few", "more", "most", "other", "such",
    }
    words = re.findall(r'\b[a-zA-Z]{3,}\b', text)
    topic_words = [w for w in words if w.lower() not in STOPWORDS][:max_words]
    return " ".join(topic_words) if topic_words else text[:40]


def _clean_answer(text: str) -> str:
    """Trim answer to a reasonable length and ensure it ends cleanly."""
    text = text.strip()
    # Cap at ~300 chars; cut at last sentence boundary
    if len(text) > 300:
        cutoff = text.rfind(".", 0, 300)
        if cutoff > 100:
            text = text[: cutoff + 1]
        else:
            text = text[:300]
    return text


def _make_prompt(context: str, question: str, answer: str) -> str:
    """Formats a single training example as instruction-tuning text."""
    return (
        f"You are a document assistant. Use ONLY the context below.\n"
        f"If the answer is not in the context, say \"{_NEGATIVE_RESPONSE}\"\n\n"
        f"CONTEXT:\n{context}\n\n"
        f"QUESTION:\n{question}\n\n"
        f"ANSWER:\n{answer}"
    )


def build_domain_dataset(
    docs: list,          # list of LangChain Document objects (page_content + metadata)
    output_path: str,
    min_chunk_len: int = 80,
    samples_per_chunk: int = 2,
    negative_ratio: float = 0.15,
) -> int:
    """
    Creates a JSONL fine-tuning dataset from document chunks.

    Parameters
    ----------
    docs             : LangChain Document list (from rag.indexer.load_documents)
    output_path      : path to write train.jsonl
    min_chunk_len    : skip chunks shorter than this (headers, page numbers, etc.)
    samples_per_chunk: how many Q&A pairs to generate per chunk
    negative_ratio   : fraction of samples that teach "I don't know" boundary

    Returns
    -------
    int : number of training samples written
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    samples = []

    for doc in docs:
        chunk = doc.page_content.strip()
        if len(chunk) < min_chunk_len:
            continue   # skip noise chunks

        topic = _extract_topic(chunk)
        answer = _clean_answer(chunk)

        # Generate factual question variants for this chunk
        templates = _FACTUAL_TEMPLATES + _DEFINITION_TEMPLATES
        chosen = random.sample(templates, min(samples_per_chunk, len(templates)))

        for tmpl in chosen:
            question = tmpl.format(topic=topic)
            samples.append({
                "text": _make_prompt(chunk, question, answer)
            })

    # ── Negative samples (out-of-scope boundary teaching) ─────────────────────
    # Use the first real chunk as context (model must learn to ignore it for OOS)
    if docs:
        anchor_chunk = docs[0].page_content.strip()[:400]
        n_negatives = max(1, int(len(samples) * negative_ratio))
        neg_topics = random.choices(_NEGATIVE_TOPICS, k=n_negatives)
        for t in neg_topics:
            question = f"What is {t}?"
            samples.append({
                "text": _make_prompt(anchor_chunk, question, _NEGATIVE_RESPONSE)
            })

    # ── Shuffle and write ──────────────────────────────────────────────────────
    random.shuffle(samples)

    with open(output_path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"[dataset] {len(samples)} training samples → {output_path}")
    return len(samples)
