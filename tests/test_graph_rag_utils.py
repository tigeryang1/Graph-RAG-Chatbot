from graph_rag_utils import build_graph_context, extract_terms


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
