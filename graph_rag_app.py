from __future__ import annotations

import os
from pathlib import Path
from typing import List, Tuple

import streamlit as st
from dotenv import dotenv_values, load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from graph_rag_utils import (
    build_graph_context,
    build_llm,
    get_neo4j_settings,
    mask_secret,
    query_graph,
)


APP_DIR = Path(__file__).resolve().parent
ENV_PATH = APP_DIR / ".env"

load_dotenv(dotenv_path=ENV_PATH)

DEFAULT_SYSTEM_PROMPT = (
    "You are a Graph RAG assistant. Use the retrieved Neo4j graph context to answer the question. "
    "If the graph evidence is weak or missing, say that clearly instead of guessing."
)


def get_api_key_status() -> tuple[str, str | None]:
    env_value = os.environ.get("GEMINI_API_KEY")
    if env_value:
        return "environment", env_value
    file_value = dotenv_values(ENV_PATH).get("GEMINI_API_KEY")
    if file_value:
        return ".env file present but not loaded", file_value
    return "not found", None


def init_state() -> None:
    if "graph_chat_history" not in st.session_state:
        st.session_state.graph_chat_history: List[Tuple[str, str]] = []
    if "graph_system_prompt" not in st.session_state:
        st.session_state.graph_system_prompt = DEFAULT_SYSTEM_PROMPT
    if "graph_last_context" not in st.session_state:
        st.session_state.graph_last_context = ""
    if "graph_last_rows" not in st.session_state:
        st.session_state.graph_last_rows = []
    if "graph_last_cypher" not in st.session_state:
        st.session_state.graph_last_cypher = ""


def build_messages(
    history: List[Tuple[str, str]],
    system_prompt: str,
    question: str,
    graph_context: str,
) -> List[BaseMessage]:
    messages: List[BaseMessage] = [
        SystemMessage(
            content=(
                f"{system_prompt}\n\n"
                "Retrieved graph context:\n"
                f"{graph_context}\n\n"
                "Use the graph context as the primary source of truth."
            )
        )
    ]
    for role, text in history:
        if role == "user":
            messages.append(HumanMessage(content=text))
        else:
            messages.append(AIMessage(content=text))
    messages.append(HumanMessage(content=question))
    return messages


def sidebar() -> tuple[str, float, int]:
    st.sidebar.title("Graph RAG Settings")
    model_name = st.sidebar.selectbox(
        "Gemini model",
        options=[
            os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
            "gemini-2.0-flash-lite",
            "gemini-2.0-flash",
        ],
        index=0,
    )
    temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=0.2, step=0.1)
    row_limit = st.sidebar.slider("Graph rows", min_value=3, max_value=20, value=8, step=1)
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
    st.sidebar.write(f"Neo4j URI: `{neo4j_settings['uri']}`")
    st.sidebar.write(f"Neo4j user: `{neo4j_settings['username']}`")
    st.sidebar.write(f"Neo4j password: `{mask_secret(neo4j_settings['password'])}`")
    return model_name, temperature, row_limit


def render_history() -> None:
    for role, text in st.session_state.graph_chat_history:
        with st.chat_message("user" if role == "user" else "assistant"):
            st.markdown(text)


def render_context_panel() -> None:
    with st.expander("Last graph retrieval", expanded=False):
        if not st.session_state.graph_last_context:
            st.caption("No graph retrieval yet.")
            return
        st.code(st.session_state.graph_last_cypher, language="cypher")
        st.text(st.session_state.graph_last_context)


def main() -> None:
    st.set_page_config(page_title="Gemini Graph RAG Chatbot", page_icon=":spider_web:", layout="centered")
    init_state()

    st.title("Gemini Graph RAG Chatbot")
    st.caption("Streamlit + Gemini + local Neo4j graph retrieval")
    st.write(
        "This sample queries a local Neo4j database, turns graph matches into textual context, "
        "and then asks Gemini to answer based on that graph evidence."
    )

    model_name, temperature, row_limit = sidebar()
    render_history()
    render_context_panel()

    prompt = st.chat_input("Ask a graph question...")
    if not prompt:
        return

    st.session_state.graph_chat_history.append(("user", prompt))
    with st.chat_message("user"):
        st.markdown(prompt)

    try:
        retrieval = query_graph(prompt, limit=row_limit)
        graph_context = build_graph_context(retrieval["rows"])
        st.session_state.graph_last_rows = retrieval["rows"]
        st.session_state.graph_last_cypher = retrieval["cypher"]
        st.session_state.graph_last_context = graph_context

        llm = build_llm(model_name=model_name, temperature=temperature)
        messages = build_messages(
            history=st.session_state.graph_chat_history[:-1],
            system_prompt=st.session_state.graph_system_prompt,
            question=prompt,
            graph_context=graph_context,
        )
        response = llm.invoke(messages)
        answer = response.content if isinstance(response.content, str) else str(response.content)
    except Exception as exc:
        answer = f"Request failed: {exc}"

    st.session_state.graph_chat_history.append(("assistant", answer))
    with st.chat_message("assistant"):
        st.markdown(answer)


if __name__ == "__main__":
    main()
