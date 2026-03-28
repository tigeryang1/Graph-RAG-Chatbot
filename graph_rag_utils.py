from __future__ import annotations

import os
import re
from typing import Any

from dotenv import load_dotenv

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
except ModuleNotFoundError:  # pragma: no cover - optional until installed
    ChatGoogleGenerativeAI = None

try:
    from neo4j import GraphDatabase
except ModuleNotFoundError:  # pragma: no cover - optional until installed
    GraphDatabase = None


load_dotenv()

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


GRAPH_SEARCH_CYPHER = """
MATCH (n)
WHERE any(term in $terms WHERE
    any(key in keys(n) WHERE toLower(toString(n[key])) CONTAINS term)
)
OPTIONAL MATCH (n)-[r]-(m)
RETURN
    labels(n) AS source_labels,
    properties(n) AS source_props,
    type(r) AS rel_type,
    labels(m) AS target_labels,
    properties(m) AS target_props
LIMIT $limit
""".strip()


def mask_secret(value: str | None) -> str:
    if not value:
        return "not set"
    if len(value) <= 8:
        return "*" * len(value)
    return f"{value[:4]}...{value[-4:]}"


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


def build_llm(model_name: str, temperature: float) -> Any:
    if ChatGoogleGenerativeAI is None:
        raise ModuleNotFoundError("langchain-google-genai is required to run the Graph RAG app.")

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("Missing GEMINI_API_KEY. Set it in your environment or a .env file.")

    return ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=api_key,
        temperature=temperature,
    )


def get_neo4j_settings() -> dict[str, str | None]:
    return {
        "uri": os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        "username": os.getenv("NEO4J_USERNAME", "neo4j"),
        "password": os.getenv("NEO4J_PASSWORD"),
    }


def query_graph(question: str, limit: int = 8) -> dict[str, Any]:
    if GraphDatabase is None:
        raise ModuleNotFoundError("neo4j is required to query the local graph.")

    settings = get_neo4j_settings()
    if not settings["password"]:
        raise ValueError("Missing NEO4J_PASSWORD. Set it in your environment or a .env file.")

    terms = extract_terms(question)
    if not terms:
        return {"terms": [], "rows": [], "cypher": GRAPH_SEARCH_CYPHER}

    driver = GraphDatabase.driver(
        settings["uri"],
        auth=(settings["username"], settings["password"]),
    )
    try:
        with driver.session() as session:
            result = session.run(GRAPH_SEARCH_CYPHER, terms=terms, limit=limit)
            rows = [record.data() for record in result]
    finally:
        driver.close()

    return {"terms": terms, "rows": rows, "cypher": GRAPH_SEARCH_CYPHER}


def _format_props(props: dict[str, Any] | None) -> str:
    if not props:
        return "{}"
    items = ", ".join(f"{key}={value}" for key, value in list(props.items())[:8])
    return "{" + items + "}"


def build_graph_context(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "No graph matches found."

    lines: list[str] = []
    for index, row in enumerate(rows, start=1):
        source_labels = ":".join(row.get("source_labels") or [])
        target_labels = ":".join(row.get("target_labels") or [])
        source_props = _format_props(row.get("source_props"))
        target_props = _format_props(row.get("target_props"))
        rel_type = row.get("rel_type") or "NO_RELATION"
        lines.append(
            f"{index}. ({source_labels} {source_props}) -[{rel_type}]- ({target_labels} {target_props})"
        )
    return "\n".join(lines)
