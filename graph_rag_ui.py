import os

import streamlit as st

from graph_rag_auth import get_api_key_status, mask_secret
from graph_rag_config import DEFAULT_EMBEDDING_MODEL, DEFAULT_EVIDENCE_MODE, DEFAULT_RETRIEVAL_MODE
from graph_rag_demo import DEMO_OVERVIEW, DEMO_PROMPTS
from graph_retrieval import get_neo4j_settings
from graph_rag_llm import get_available_models, parse_model_chain


def sidebar() -> tuple[str, list[str], float, int, str, str]:
    st.sidebar.title("Customer 360 Settings")
    available_models = get_available_models()
    configured_primary = os.getenv("GEMINI_MODEL", available_models[0])
    if configured_primary not in available_models:
        available_models = [configured_primary, *available_models]
    model_name = st.sidebar.selectbox(
        "Gemini model",
        options=available_models,
        index=available_models.index(configured_primary),
    )
    default_fallbacks = [
        model for model in parse_model_chain(model_name)[1:] if model in available_models
    ]
    fallback_models = st.sidebar.multiselect(
        "Fallback models",
        options=[model for model in available_models if model != model_name],
        default=default_fallbacks,
        help="These models are tried automatically if the primary model hits a quota or rate limit.",
    )
    retrieval_mode = st.sidebar.selectbox(
        "Retrieval mode",
        options=["Fixed query", "Dynamic Cypher (safe)"],
        index=["Fixed query", "Dynamic Cypher (safe)"].index(
            st.session_state.graph_retrieval_mode or DEFAULT_RETRIEVAL_MODE
        ),
        help=(
            "Fixed query uses the built-in keyword graph query. Dynamic Cypher asks Gemini to generate a "
            "read-only Cypher query, then validates it locally before execution."
        ),
    )
    st.session_state.graph_retrieval_mode = retrieval_mode
    evidence_mode = st.sidebar.selectbox(
        "Evidence mode",
        options=["Graph + Notes", "Graph only", "Notes only"],
        index=["Graph + Notes", "Graph only", "Notes only"].index(
            st.session_state.graph_evidence_mode or DEFAULT_EVIDENCE_MODE
        ),
        help="Choose whether the answer should use Neo4j graph evidence, local vector-note RAG, or both.",
    )
    st.session_state.graph_evidence_mode = evidence_mode
    temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=0.2, step=0.1)
    row_limit = st.sidebar.slider("Relationship rows", min_value=3, max_value=20, value=8, step=1)
    st.session_state.graph_system_prompt = st.sidebar.text_area(
        "System prompt",
        value=st.session_state.graph_system_prompt,
        height=140,
    )
    if st.sidebar.button("Clear chat"):
        st.session_state.graph_chat_history = []
        st.session_state.graph_last_context = ""
        st.session_state.graph_last_rows = []
        st.session_state.graph_last_cypher = ""
        st.session_state.graph_last_doc_hits = []
        st.session_state.graph_last_doc_context = ""
        st.session_state.graph_last_graph_help = ""
        st.session_state.graph_last_model = ""
        st.session_state.graph_model_failovers = []
        st.session_state.graph_last_cypher_model = ""
        st.session_state.graph_cypher_failovers = []
        st.rerun()

    source, key = get_api_key_status()
    neo4j_settings = get_neo4j_settings()
    st.sidebar.divider()
    with st.sidebar.expander("Diagnostics", expanded=False):
        st.write(f"Gemini key source: `{source}`")
        st.write(f"Gemini key detected: `{mask_secret(key)}`")
        st.write(f"Model chain: `{', '.join(parse_model_chain(model_name, fallback_models))}`")
        st.write(f"Retrieval mode: `{retrieval_mode}`")
        st.write(f"Evidence mode: `{evidence_mode}`")
        st.write(f"Note embedding model: `{DEFAULT_EMBEDDING_MODEL}`")
        st.write("Note index: `Local in-memory FAISS`")
        st.write(f"Neo4j URI: `{neo4j_settings['uri']}`")
        st.write(f"Neo4j user: `{neo4j_settings['username']}`")
        st.write(f"Neo4j password: `{mask_secret(neo4j_settings['password'])}`")
    return model_name, fallback_models, temperature, row_limit, retrieval_mode, evidence_mode


def render_history() -> None:
    for role, text in st.session_state.graph_chat_history:
        with st.chat_message("user" if role == "user" else "assistant"):
            st.markdown(text)


def render_demo_panel() -> None:
    has_history = bool(st.session_state.graph_chat_history)
    with st.expander("Demo Guide", expanded=not has_history):
        st.write(DEMO_OVERVIEW)
        st.write("Suggested prompts:")
        for item in DEMO_PROMPTS:
            cols = st.columns([4, 1])
            with cols[0]:
                st.markdown(f"**{item['title']}**")
                st.caption(item["why"])
                st.code(item["prompt"])
            with cols[1]:
                if st.button("Try it", key=f"demo-{item['title']}"):
                    st.session_state.graph_demo_prompt = item["prompt"]
                    st.rerun()


def render_evidence_overview() -> None:
    if (
        not st.session_state.graph_last_context
        and not st.session_state.graph_last_doc_hits
        and not st.session_state.graph_last_doc_context
    ):
        return

    st.subheader("Evidence used for the latest answer")
    files_used = st.session_state.graph_last_doc_hits
    file_count = len(files_used)
    row_count = len(st.session_state.graph_last_rows)
    metric_cols = st.columns(3)
    metric_cols[0].metric("Evidence mode", st.session_state.graph_evidence_mode)
    metric_cols[1].metric("Graph rows", row_count)
    metric_cols[2].metric("Note chunks", file_count)

    left_col, right_col = st.columns([1.3, 1.0])

    with left_col:
        st.markdown("**Note chunks used in RAG**")
        if st.session_state.graph_evidence_mode == "Graph only":
            st.caption("Notes retrieval is disabled in Graph only mode.")
        elif not files_used:
            st.caption("No note chunks matched this question.")
        else:
            for item in files_used:
                st.markdown(f"- `{item['name']}`")
                st.caption(f"Retrieved chunk: {item['snippet']}")

    with right_col:
        st.markdown("**How the graph helped**")
        if st.session_state.graph_evidence_mode == "Notes only":
            st.caption("Graph retrieval is disabled in Notes only mode.")
        else:
            st.write(st.session_state.graph_last_graph_help or "No graph contribution summary available.")
        if st.session_state.graph_last_doc_context and st.session_state.graph_evidence_mode != "Graph only":
            st.markdown("**Combined vector-note context**")
            st.caption(st.session_state.graph_last_doc_context)


def render_context_panel() -> None:
    with st.expander("Last Customer 360 retrieval", expanded=False):
        if (
            not st.session_state.graph_last_context
            and not st.session_state.graph_last_doc_context
            and not st.session_state.graph_last_cypher
        ):
            st.caption("No retrieval yet.")
        else:
            st.write(f"Evidence mode: `{st.session_state.graph_evidence_mode}`")
            st.write(f"Retrieval mode: `{st.session_state.graph_retrieval_mode}`")
            st.code(st.session_state.graph_last_cypher, language="cypher")
            if st.session_state.graph_evidence_mode != "Notes only":
                st.text(st.session_state.graph_last_context)
            if st.session_state.graph_evidence_mode != "Graph only":
                st.markdown("**Supporting vector-note context**")
                st.caption(st.session_state.graph_last_doc_context)

    with st.expander("Last model run", expanded=False):
        if not st.session_state.graph_last_model:
            st.caption("No model invocation yet.")
            return
        st.write(f"Model used: `{st.session_state.graph_last_model}`")
        if st.session_state.graph_model_failovers:
            st.write("Automatic fallback attempts:")
            for item in st.session_state.graph_model_failovers:
                st.code(item)
        if st.session_state.graph_last_cypher_model:
            st.write(f"Cypher generation model: `{st.session_state.graph_last_cypher_model}`")
        if st.session_state.graph_cypher_failovers:
            st.write("Cypher generation fallback attempts:")
            for item in st.session_state.graph_cypher_failovers:
                st.code(item)
