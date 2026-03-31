import re


WORD_RE = re.compile(r"[a-zA-Z0-9_]+")
STOPWORDS = {
    "what",
    "does",
    "about",
    "with",
    "from",
    "that",
    "this",
    "have",
    "into",
    "your",
    "their",
    "there",
    "where",
    "when",
    "which",
    "would",
    "could",
    "should",
    "and",
    "the",
}


def extract_terms(question: str, max_terms: int = 6) -> list[str]:
    seen: list[str] = []
    for token in WORD_RE.findall(question.lower()):
        if len(token) < 3:
            continue
        if token in STOPWORDS:
            continue
        if token not in seen:
            seen.append(token)
        if len(seen) >= max_terms:
            break
    return seen
