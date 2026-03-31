from typing import List, Tuple

import streamlit as st
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from graph_rag_config import DEFAULT_EVIDENCE_MODE, DEFAULT_RETRIEVAL_MODE, DEFAULT_SYSTEM_PROMPT


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
    if "graph_last_doc_hits" not in st.session_state:
        st.session_state.graph_last_doc_hits = []
    if "graph_last_doc_context" not in st.session_state:
        st.session_state.graph_last_doc_context = ""
    if "graph_last_graph_help" not in st.session_state:
        st.session_state.graph_last_graph_help = ""
    if "graph_last_model" not in st.session_state:
        st.session_state.graph_last_model = ""
    if "graph_model_failovers" not in st.session_state:
        st.session_state.graph_model_failovers = []
    if "graph_retrieval_mode" not in st.session_state:
        st.session_state.graph_retrieval_mode = DEFAULT_RETRIEVAL_MODE
    if "graph_evidence_mode" not in st.session_state:
        st.session_state.graph_evidence_mode = DEFAULT_EVIDENCE_MODE
    if "graph_last_cypher_model" not in st.session_state:
        st.session_state.graph_last_cypher_model = ""
    if "graph_cypher_failovers" not in st.session_state:
        st.session_state.graph_cypher_failovers = []
    if "graph_demo_prompt" not in st.session_state:
        st.session_state.graph_demo_prompt = ""


def build_messages(
    history: List[Tuple[str, str]],
    system_prompt: str,
    question: str,
    graph_context: str,
    document_context: str,
    evidence_mode: str,
) -> List[BaseMessage]:
    messages: List[BaseMessage] = [
        SystemMessage(
            content=(
                f"{system_prompt}\n\n"
                f"Evidence mode: {evidence_mode}\n\n"
                "Retrieved customer graph context:\n"
                f"{graph_context}\n\n"
                "Retrieved supporting account notes:\n"
                f"{document_context}\n\n"
                "Use the active evidence sources as the basis for the answer. "
                "When graph evidence is present, treat it as the primary source of truth."
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
