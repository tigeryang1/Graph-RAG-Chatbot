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

DEFAULT_MODEL_OPTIONS = [
    "Gemini 3.1 Flash Lite",
    "Gemini 3 Flash",
    "Gemini 2.5 Flash",
    "Gemini 2.5 Flash Lite",
]
MODEL_NAME_ALIASES = {
    "Gemini 3.1 Flash Lite": "gemini-3.1-flash-lite-preview",
    "Gemini 3 Flash": "gemini-3-flash-preview",
    "Gemini 2.5 Flash": "gemini-2.5-flash",
    "Gemini 2.5 Flash Lite": "gemini-2.5-flash-lite",
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
        model=MODEL_NAME_ALIASES.get(model_name, model_name),
        google_api_key=api_key,
        temperature=temperature,
    )


def get_available_models() -> list[str]:
    env_value = os.getenv("GEMINI_AVAILABLE_MODELS", "")
    configured = [item.strip() for item in env_value.split(",") if item.strip()]
    models = configured or DEFAULT_MODEL_OPTIONS
    return list(dict.fromkeys(models))


def parse_model_chain(primary_model: str, fallback_models: list[str] | None = None) -> list[str]:
    configured = fallback_models or []
    if not configured:
        env_value = os.getenv("GEMINI_FALLBACK_MODELS", "")
        configured = [item.strip() for item in env_value.split(",") if item.strip()]

    chain: list[str] = []
    for candidate in [primary_model, *configured, *get_available_models()]:
        if candidate and candidate not in chain:
            chain.append(candidate)
    return chain


def is_model_limit_error(exc: Exception) -> bool:
    message = str(exc).lower()
    signals = [
        "429",
        "quota",
        "rate limit",
        "resource exhausted",
        "resource_exhausted",
        "too many requests",
        "exceeded",
        "limit reached",
        "invalid_argument",
        "unexpected model name format",
        "model not found",
        "unsupported model",
    ]
    return any(signal in message for signal in signals)


def invoke_with_model_fallback(
    messages: list[Any],
    model_chain: list[str],
    temperature: float,
    llm_builder=build_llm,
) -> tuple[Any, str, list[str]]:
    if not model_chain:
        raise ValueError("Model chain is empty.")

    errors: list[str] = []
    last_exc: Exception | None = None
    for index, model_name in enumerate(model_chain):
        try:
            llm = llm_builder(model_name=model_name, temperature=temperature)
            response = llm.invoke(messages)
            return response, model_name, errors
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if index == len(model_chain) - 1 or not is_model_limit_error(exc):
                raise
            errors.append(f"{model_name}: {exc}")

    if last_exc is not None:
        raise last_exc
    raise RuntimeError("Model invocation failed without an exception.")


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
