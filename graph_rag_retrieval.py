import os
import re
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

try:
    from neo4j import GraphDatabase
except ModuleNotFoundError:  # pragma: no cover - optional until installed
    GraphDatabase = None


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

ACCOUNT_CENTERED_CYPHER = """
MATCH (a:Account)
WHERE any(term in $terms WHERE
    any(key in keys(a) WHERE toLower(toString(a[key])) CONTAINS term)
)
MATCH p = (a)-[*1..2]-(x)
WHERE all(rel in relationships(p) WHERE type(rel) IN ['WORKS_FOR', 'FOR_ACCOUNT', 'OWNS', 'TARGETS', 'INFLUENCED'])
  AND all(node in nodes(p) WHERE any(label in labels(node) WHERE label IN ['Account', 'Contact', 'Opportunity', 'Case', 'Campaign', 'User']))
UNWIND range(0, size(relationships(p)) - 1) AS idx
WITH nodes(p)[idx] AS n, relationships(p)[idx] AS r, nodes(p)[idx + 1] AS m
RETURN DISTINCT
    labels(n) AS source_labels,
    properties(n) AS source_props,
    type(r) AS rel_type,
    labels(m) AS target_labels,
    properties(m) AS target_props
LIMIT $limit
""".strip()

ALLOWED_NODE_LABELS = {"Account", "Contact", "Opportunity", "Case", "Campaign", "User"}
ALLOWED_REL_TYPES = {"WORKS_FOR", "FOR_ACCOUNT", "OWNS", "TARGETS", "INFLUENCED"}
DANGEROUS_CYPHER_PATTERNS = [
    "CREATE",
    "MERGE",
    "DELETE",
    "DETACH",
    "SET ",
    "REMOVE",
    "DROP",
    "LOAD CSV",
    "CALL ",
    "APOC",
    "DBMS",
    "FOREACH",
]

SAFE_DYNAMIC_CYPHER_PROMPT = """
You generate a single safe, read-only Cypher query for a Customer 360 Neo4j graph.

Allowed node labels:
- Account
- Contact
- Opportunity
- Case
- Campaign
- User

Allowed relationship types:
- WORKS_FOR
- FOR_ACCOUNT
- OWNS
- TARGETS
- INFLUENCED

Requirements:
- Return only Cypher. No explanation. No markdown fences.
- Read-only query only.
- Do not use CREATE, MERGE, DELETE, DETACH, SET, REMOVE, DROP, LOAD CSV, CALL, APOC, DBMS, or FOREACH.
- Return exactly these columns:
  labels(n) AS source_labels,
  properties(n) AS source_props,
  type(r) AS rel_type,
  labels(m) AS target_labels,
  properties(m) AS target_props
- Use variables n, r, m for the returned edge rows.
- Prefer account-centric retrieval when the question mentions an account or customer name.
- Limit results to at most {limit} rows.
""".strip()

KNOWLEDGE_BASE_DIR = Path(__file__).resolve().parent / "knowledge_base"


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


def get_neo4j_settings() -> dict[str, str | None]:
    return {
        "uri": os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        "username": os.getenv("NEO4J_USERNAME", "neo4j"),
        "password": os.getenv("NEO4J_PASSWORD"),
    }


def _run_cypher(cypher: str, **params: Any) -> list[dict[str, Any]]:
    if GraphDatabase is None:
        raise ModuleNotFoundError("neo4j is required to query the local graph.")

    settings = get_neo4j_settings()
    if not settings["password"]:
        raise ValueError("Missing NEO4J_PASSWORD. Set it in your environment or a .env file.")

    driver = GraphDatabase.driver(
        settings["uri"],
        auth=(settings["username"], settings["password"]),
    )
    try:
        with driver.session() as session:
            result = session.run(cypher, **params)
            rows = [record.data() for record in result]
    finally:
        driver.close()

    return rows


def query_graph(question: str, limit: int = 8) -> dict[str, Any]:
    terms = extract_terms(question)
    if not terms:
        return {"terms": [], "rows": [], "cypher": GRAPH_SEARCH_CYPHER}

    account_rows = _run_cypher(ACCOUNT_CENTERED_CYPHER, terms=terms, limit=limit)
    if account_rows:
        return {"terms": terms, "rows": account_rows, "cypher": ACCOUNT_CENTERED_CYPHER}

    rows = _run_cypher(GRAPH_SEARCH_CYPHER, terms=terms, limit=limit)
    return {"terms": terms, "rows": rows, "cypher": GRAPH_SEARCH_CYPHER}


def build_dynamic_cypher_messages(question: str, limit: int) -> list[Any]:
    return [
        SystemMessage(content=SAFE_DYNAMIC_CYPHER_PROMPT.format(limit=limit)),
        HumanMessage(content=question),
    ]


def _strip_cypher_fences(text: str) -> str:
    cleaned = text.strip()
    fence_match = re.match(r"^```(?:cypher)?\s*(.*?)\s*```$", cleaned, flags=re.IGNORECASE | re.DOTALL)
    if fence_match:
        return fence_match.group(1).strip()
    return cleaned


def _extract_node_labels(cypher: str) -> set[str]:
    labels: set[str] = set()
    for node_body in re.findall(r"\(([^)]*)\)", cypher):
        labels.update(re.findall(r":`?([A-Z][A-Za-z0-9_]*)`?", node_body))
    return labels


def _extract_rel_types(cypher: str) -> set[str]:
    rel_types: set[str] = set()
    for rel_body in re.findall(r"\[([^\]]*)\]", cypher):
        rel_types.update(re.findall(r":`?([A-Z][A-Z0-9_]*)`?", rel_body))
    return rel_types


def validate_and_prepare_cypher(raw_cypher: str, limit: int) -> str:
    cypher = _strip_cypher_fences(raw_cypher).rstrip(";").strip()
    if not cypher:
        raise ValueError("Dynamic Cypher generation returned an empty query.")

    upper = re.sub(r"\s+", " ", cypher).upper()
    if ";" in cypher:
        raise ValueError("Dynamic Cypher must contain a single statement.")
    if "RETURN" not in upper:
        raise ValueError("Dynamic Cypher must include a RETURN clause.")
    if not upper.startswith(("MATCH ", "OPTIONAL MATCH ", "WITH ", "UNWIND ")):
        raise ValueError("Dynamic Cypher must start with a read-only MATCH-style clause.")
    if any(pattern in upper for pattern in DANGEROUS_CYPHER_PATTERNS):
        raise ValueError("Dynamic Cypher contains unsupported write or procedure operations.")

    labels = _extract_node_labels(cypher)
    if not labels.issubset(ALLOWED_NODE_LABELS):
        raise ValueError(f"Dynamic Cypher uses unsupported labels: {sorted(labels - ALLOWED_NODE_LABELS)}")

    rel_types = _extract_rel_types(cypher)
    if not rel_types.issubset(ALLOWED_REL_TYPES):
        raise ValueError(
            f"Dynamic Cypher uses unsupported relationship types: {sorted(rel_types - ALLOWED_REL_TYPES)}"
        )

    required_aliases = [
        "source_labels",
        "source_props",
        "rel_type",
        "target_labels",
        "target_props",
    ]
    lowered = cypher.lower()
    if not all(alias in lowered for alias in required_aliases):
        raise ValueError("Dynamic Cypher must return the required alias set for graph rendering.")

    limit_match = re.search(r"\bLIMIT\s+(\d+)\b", cypher, flags=re.IGNORECASE)
    if limit_match:
        existing_limit = int(limit_match.group(1))
        safe_limit = min(existing_limit, limit)
        cypher = re.sub(
            r"\bLIMIT\s+\d+\b",
            f"LIMIT {safe_limit}",
            cypher,
            count=1,
            flags=re.IGNORECASE,
        )
    else:
        cypher = f"{cypher}\nLIMIT {limit}"

    return cypher


def execute_dynamic_cypher(cypher: str) -> list[dict[str, Any]]:
    return _run_cypher(cypher)


def _format_props(props: dict[str, Any] | None) -> str:
    if not props:
        return "{}"
    important_keys = ["name", "subject", "title", "stage", "status", "priority", "email", "owner"]
    ordered: list[tuple[str, Any]] = []
    for key in important_keys:
        if key in props:
            ordered.append((key, props[key]))
    for key, value in props.items():
        if key not in {item[0] for item in ordered}:
            ordered.append((key, value))
        if len(ordered) >= 8:
            break
    items = ", ".join(f"{key}={value}" for key, value in ordered[:8])
    return "{" + items + "}"


def build_graph_context(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "No customer relationship matches found."

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


def retrieve_supporting_docs(question: str, max_docs: int = 3) -> list[dict[str, str | int]]:
    if not KNOWLEDGE_BASE_DIR.exists():
        return []

    terms = extract_terms(question, max_terms=8)
    if not terms:
        return []

    hits: list[dict[str, str | int]] = []
    for path in sorted(KNOWLEDGE_BASE_DIR.glob("*.txt")):
        content = path.read_text(encoding="utf-8")
        lower_content = content.lower()
        lower_name = path.stem.lower()
        score = 0
        for term in terms:
            score += lower_content.count(term)
            if term in lower_name:
                score += 2
        if score <= 0:
            continue

        snippet = ""
        for line in content.splitlines():
            if any(term in line.lower() for term in terms):
                snippet = line.strip()
                break
        if not snippet:
            snippet = " ".join(content.split())[:180]

        hits.append(
            {
                "name": path.name,
                "path": str(path),
                "score": score,
                "snippet": snippet[:220],
            }
        )

    hits.sort(key=lambda item: int(item["score"]), reverse=True)
    return hits[:max_docs]


def build_document_context(doc_hits: list[dict[str, str | int]]) -> str:
    if not doc_hits:
        return "No supporting account notes matched the question."

    lines = []
    for item in doc_hits:
        lines.append(f"[{item['name']}] {item['snippet']}")
    return "\n".join(lines)


def summarize_graph_contribution(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "The graph did not add relationship evidence for this question."

    rel_types = sorted({row.get("rel_type") for row in rows if row.get("rel_type")})
    entity_counts = {
        label: 0
        for label in ["Account", "Contact", "Opportunity", "Case", "Campaign", "User"]
    }
    seen_entities: set[tuple[str, str]] = set()

    for row in rows:
        for side in ("source", "target"):
            labels = row.get(f"{side}_labels") or []
            props = row.get(f"{side}_props") or {}
            entity_id = str(props.get("id") or props.get("name") or props.get("subject") or "")
            for label in labels:
                key = (label, entity_id)
                if label in entity_counts and entity_id and key not in seen_entities:
                    entity_counts[label] += 1
                    seen_entities.add(key)

    populated = [f"{label}: {count}" for label, count in entity_counts.items() if count]
    rel_text = ", ".join(rel_types) if rel_types else "no explicit relationship types"
    entity_text = ", ".join(populated) if populated else "no entity counts"
    return f"The graph supplied relationship evidence across {entity_text}. Relationship types seen: {rel_text}."
