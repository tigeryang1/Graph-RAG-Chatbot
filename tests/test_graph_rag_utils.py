from types import SimpleNamespace

from graph_rag_utils import (
    build_graph_context,
    extract_terms,
    get_available_models,
    invoke_with_model_fallback,
    is_model_limit_error,
    parse_model_chain,
    MODEL_NAME_ALIASES,
)


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
    assert build_graph_context([]) == "No graph matches found."


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
