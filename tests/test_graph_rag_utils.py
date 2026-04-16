from types import SimpleNamespace

import pytest

from graph_rag_llm import (
    MODEL_NAME_ALIASES,
    extract_response_text,
    get_available_models,
    invoke_with_model_fallback,
    is_model_limit_error,
    parse_model_chain,
    uses_structured_content_blocks,
)
from graph_retrieval import (
    ACCOUNT_CENTERED_CYPHER,
    build_graph_context,
    build_dynamic_cypher_messages,
    summarize_graph_contribution,
    validate_and_prepare_cypher,
)
from rag_notes import build_document_context, retrieve_supporting_docs
from common_terms import extract_terms


def test_extract_terms_filters_short_tokens() -> None:
    terms = extract_terms("What does CRM graph say about Nike and ads?")
    assert "crm" in terms
    assert "ads" in terms
    assert "do" not in terms
    assert "what" not in terms


def test_build_graph_context_formats_rows() -> None:
    context = build_graph_context(
        [
            {
                "source_labels": ["Company"],
                "source_props": {"name": "Acme"},
                "rel_type": "OWNS",
                "target_labels": ["Product"],
                "target_props": {"name": "Atlas"},
            }
        ]
    )
    assert "Company" in context
    assert "OWNS" in context
    assert "Atlas" in context


def test_build_graph_context_handles_empty_rows() -> None:
    assert build_graph_context([]) == "No customer relationship matches found."


def test_parse_model_chain_deduplicates_primary_and_fallbacks() -> None:
    chain = parse_model_chain(
        "Gemini 3.1 Flash Lite",
        ["Gemini 3 Flash", "Gemini 3.1 Flash Lite", "Gemini 2.5 Flash"],
    )
    assert chain == ["Gemini 3.1 Flash Lite", "Gemini 3 Flash", "Gemini 2.5 Flash", "Gemini 2.5 Flash Lite"]


def test_get_available_models_uses_env(monkeypatch) -> None:
    monkeypatch.setenv(
        "GEMINI_AVAILABLE_MODELS",
        "Gemini 3.1 Flash Lite,Gemini 3 Flash,Gemini 2.5 Flash,Gemini 2.5 Flash Lite",
    )
    assert get_available_models() == [
        "Gemini 3.1 Flash Lite",
        "Gemini 3 Flash",
        "Gemini 2.5 Flash",
        "Gemini 2.5 Flash Lite",
    ]


def test_is_model_limit_error_matches_quota_signals() -> None:
    assert is_model_limit_error(Exception("429 RESOURCE_EXHAUSTED: quota exceeded")) is True
    assert is_model_limit_error(Exception("400 INVALID_ARGUMENT: unexpected model name format")) is True
    assert is_model_limit_error(Exception("Invalid API key")) is False


def test_model_aliases_include_verified_preview_and_flash_ids() -> None:
    assert MODEL_NAME_ALIASES["Gemini 3.1 Flash Lite"] == "gemini-3.1-flash-lite-preview"
    assert MODEL_NAME_ALIASES["Gemini 3 Flash"] == "gemini-3-flash-preview"


def test_uses_structured_content_blocks_matches_gemini_31_flash_lite() -> None:
    assert uses_structured_content_blocks("Gemini 3.1 Flash Lite") is True
    assert uses_structured_content_blocks("gemini-3.1-flash-lite-preview") is True
    assert uses_structured_content_blocks("Gemini 3 Flash") is False


def test_invoke_with_model_fallback_switches_on_limit_error() -> None:
    attempts: list[str] = []

    def fake_builder(model_name: str, temperature: float):
        class FakeLlm:
            def invoke(self, messages):
                attempts.append(model_name)
                if model_name == "Gemini 3.1 Flash Lite":
                    raise Exception("429 RESOURCE_EXHAUSTED: quota exceeded")
                return SimpleNamespace(content=f"response from {model_name}")

        return FakeLlm()

    response, used_model, errors = invoke_with_model_fallback(
        messages=["hello"],
        model_chain=["Gemini 3.1 Flash Lite", "Gemini 3 Flash"],
        temperature=0.2,
        llm_builder=fake_builder,
    )

    assert attempts == ["Gemini 3.1 Flash Lite", "Gemini 3 Flash"]
    assert used_model == "Gemini 3 Flash"
    assert response.content == "response from Gemini 3 Flash"
    assert len(errors) == 1


def test_extract_response_text_reads_structured_content_blocks() -> None:
    response = SimpleNamespace(
        content=[
            {
                "type": "text",
                "text": "Summary for Bright Foods.",
                "extras": {"signature": "abc"},
            },
            {
                "type": "text",
                "text": "There is one pending support case.",
            },
        ]
    )

    assert extract_response_text(response, model_name="Gemini 3.1 Flash Lite") == (
        "Summary for Bright Foods.\n\nThere is one pending support case."
    )


def test_build_dynamic_cypher_messages_includes_schema_instruction() -> None:
    messages = build_dynamic_cypher_messages("Show me Bright Foods opportunities", limit=6)

    assert "Allowed node labels" in messages[0].content
    assert messages[1].content == "Show me Bright Foods opportunities"


def test_account_centered_cypher_reaches_two_hops_for_owners() -> None:
    assert "[*1..2]" in ACCOUNT_CENTERED_CYPHER
    assert "OWNS" in ACCOUNT_CENTERED_CYPHER
    assert "MATCH (a:Account)" in ACCOUNT_CENTERED_CYPHER


def test_validate_and_prepare_cypher_allows_safe_read_only_query() -> None:
    cypher = """
    MATCH (n:Account)-[r:TARGETS]-(m:Campaign)
    RETURN
        labels(n) AS source_labels,
        properties(n) AS source_props,
        type(r) AS rel_type,
        labels(m) AS target_labels,
        properties(m) AS target_props
    """.strip()

    prepared = validate_and_prepare_cypher(cypher, limit=5)

    assert "LIMIT 5" in prepared
    assert "MATCH (n:Account)-[r:TARGETS]-(m:Campaign)" in prepared


def test_validate_and_prepare_cypher_rejects_write_queries() -> None:
    cypher = """
    MATCH (n:Account)
    SET n.status = 'Active'
    RETURN
        labels(n) AS source_labels,
        properties(n) AS source_props,
        'NO_RELATION' AS rel_type,
        [] AS target_labels,
        {} AS target_props
    """.strip()

    with pytest.raises(ValueError, match=r"(?i)write|unsupported"):
        validate_and_prepare_cypher(cypher, limit=5)


def test_validate_and_prepare_cypher_rejects_unknown_labels() -> None:
    cypher = """
    MATCH (n:Lead)-[r:FOR_ACCOUNT]-(m:Opportunity)
    RETURN
        labels(n) AS source_labels,
        properties(n) AS source_props,
        type(r) AS rel_type,
        labels(m) AS target_labels,
        properties(m) AS target_props
    LIMIT 5
    """.strip()

    with pytest.raises(ValueError, match=r"(?i)unsupported labels"):
        validate_and_prepare_cypher(cypher, limit=5)


def test_retrieve_supporting_docs_finds_matching_account_notes() -> None:
    class FakeDoc:
        def __init__(self, name: str, path: str, content: str):
            self.page_content = content
            self.metadata = {"name": name, "path": path}

    class FakeVectorStore:
        def similarity_search_with_score(self, question: str, k: int):
            assert "Bright Foods" in question
            return [
                (
                    FakeDoc(
                        "bright_account_brief.txt",
                        "C:/tmp/bright_account_brief.txt",
                        "Bright Foods is in active renewal discussions.",
                    ),
                    1.2,
                )
            ]

    import rag_notes

    original_get = rag_notes.get_vector_store

    try:
        rag_notes.get_vector_store = lambda: FakeVectorStore()  # type: ignore[assignment]
        hits = retrieve_supporting_docs("What is the risk profile for Bright Foods?")
    finally:
        rag_notes.get_vector_store = original_get  # type: ignore[assignment]

    assert hits
    assert hits[0]["name"] == "bright_account_brief.txt"


def test_build_document_context_formats_file_snippets() -> None:
    context = build_document_context(
        [
            {
                "name": "bright_account_brief.txt",
                "path": "C:/tmp/bright_account_brief.txt",
                "score": 3,
                "snippet": "Bright Foods is in active renewal discussions.",
            }
        ]
    )

    assert "[bright_account_brief.txt]" in context
    assert "active renewal discussions" in context


def test_summarize_graph_contribution_counts_entities_and_relationships() -> None:
    summary = summarize_graph_contribution(
        [
            {
                "source_labels": ["Account"],
                "source_props": {"id": "acct_001", "name": "Bright Foods"},
                "rel_type": "FOR_ACCOUNT",
                "target_labels": ["Opportunity"],
                "target_props": {"id": "opp_001", "name": "Renewal"},
            },
            {
                "source_labels": ["User"],
                "source_props": {"id": "user_001", "name": "Riley Morgan"},
                "rel_type": "OWNS",
                "target_labels": ["Case"],
                "target_props": {"id": "case_001", "subject": "Dashboard access issue"},
            },
        ]
    )

    assert "Account: 1" in summary
    assert "Opportunity: 1" in summary
    assert "FOR_ACCOUNT" in summary
    assert "OWNS" in summary
