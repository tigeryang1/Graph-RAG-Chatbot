import os

import streamlit as st

from graph_rag_auth import get_api_key_status, mask_secret
from graph_rag_config import DEFAULT_RETRIEVAL_MODE
from graph_rag_demo import DEMO_OVERVIEW, DEMO_PROMPTS
from graph_rag_llm import get_available_models, parse_model_chain
from graph_rag_retrieval import get_neo4j_settings


def sidebar() -> tuple[str, list[str], float, int, str]:
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
        st.rerun()

    source, key = get_api_key_status()
    neo4j_settings = get_neo4j_settings()
    st.sidebar.divider()
    st.sidebar.subheader("Diagnostics")
    st.sidebar.write(f"Gemini key source: `{source}`")
    st.sidebar.write(f"Gemini key detected: `{mask_secret(key)}`")
    st.sidebar.write(f"Model chain: `{', '.join(parse_model_chain(model_name, fallback_models))}`")
    st.sidebar.write(f"Retrieval mode: `{retrieval_mode}`")
    st.sidebar.write(f"Neo4j URI: `{neo4j_settings['uri']}`")
    st.sidebar.write(f"Neo4j user: `{neo4j_settings['username']}`")
    st.sidebar.write(f"Neo4j password: `{mask_secret(neo4j_settings['password'])}`")
    return model_name, fallback_models, temperature, row_limit, retrieval_mode


def render_history() -> None:
    for role, text in st.session_state.graph_chat_history:
        with st.chat_message("user" if role == "user" else "assistant"):
            st.markdown(text)


def render_demo_panel() -> None:
    with st.expander("Demo Guide", expanded=True):
        st.write(DEMO_OVERVIEW)
        st.write("Suggested prompts:")
        for item in DEMO_PROMPTS:
            cols = st.columns([4, 1])
            with cols[0]:
                st.markdown(f"**{item['title']}**")
                st.caption(item["why"])
                st.code(item["prompt"])
            with cols[1]:
                if st.button("Load", key=f"demo-{item['title']}"):
                    st.session_state.graph_demo_prompt = item["prompt"]


def render_context_panel() -> None:
    with st.expander("Last Customer 360 retrieval", expanded=False):
        if not st.session_state.graph_last_context:
            st.caption("No graph retrieval yet.")
        else:
            st.write(f"Retrieval mode: `{st.session_state.graph_retrieval_mode}`")
            st.code(st.session_state.graph_last_cypher, language="cypher")
            st.text(st.session_state.graph_last_context)

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
