import streamlit as st

from graph_rag_llm import (
    extract_response_text,
    invoke_with_model_fallback,
    parse_model_chain,
)
from graph_rag_retrieval import (
    build_graph_context,
    build_document_context,
    build_dynamic_cypher_messages,
    execute_dynamic_cypher,
    extract_terms,
    query_graph,
    retrieve_supporting_docs,
    summarize_graph_contribution,
    validate_and_prepare_cypher,
)
from graph_rag_state import build_messages, init_state
from graph_rag_ui import (
    render_context_panel,
    render_demo_panel,
    render_evidence_overview,
    render_history,
    sidebar,
)


def main() -> None:
    st.set_page_config(page_title="Gemini Graph RAG Chatbot", page_icon=":spider_web:", layout="wide")
    init_state()

    st.title("Customer 360 Graph Assistant")
    st.caption("Streamlit + Gemini + local Neo4j customer relationship retrieval")
    st.write(
        "This app retrieves account relationship context from a local Neo4j graph and asks Gemini "
        "to summarize customer connections across contacts, opportunities, cases, campaigns, and owners."
    )

    model_name, fallback_models, temperature, row_limit, retrieval_mode, evidence_mode = sidebar()
    st.markdown(
        "Ask account questions and choose whether the answer should use local Neo4j relationship evidence, "
        "matching account-note files from `knowledge_base/`, or both."
    )

    render_demo_panel()
    render_evidence_overview()
    render_history()
    render_context_panel()

    prompt = None
    demo_prompt = st.session_state.graph_demo_prompt
    if demo_prompt:
        st.info(f"Loaded demo prompt: {demo_prompt}")
        if st.button("Run demo prompt"):
            prompt = demo_prompt
            st.session_state.graph_demo_prompt = ""

    user_prompt = st.chat_input("Ask a graph question...")
    if user_prompt:
        prompt = user_prompt

    if not prompt:
        return

    st.session_state.graph_chat_history.append(("user", prompt))
    with st.chat_message("user"):
        st.markdown(prompt)

    try:
        model_chain = parse_model_chain(model_name, fallback_models)
        st.session_state.graph_last_cypher_model = ""
        st.session_state.graph_cypher_failovers = []
        use_graph = evidence_mode in {"Graph + Notes", "Graph only"}
        use_notes = evidence_mode in {"Graph + Notes", "Notes only"}

        if use_graph:
            if retrieval_mode == "Dynamic Cypher (safe)":
                cypher_response, cypher_model, cypher_failovers = invoke_with_model_fallback(
                    messages=build_dynamic_cypher_messages(prompt, row_limit),
                    model_chain=model_chain,
                    temperature=0.0,
                )
                generated_cypher = extract_response_text(cypher_response, model_name=cypher_model)
                safe_cypher = validate_and_prepare_cypher(generated_cypher, row_limit)
                rows = execute_dynamic_cypher(safe_cypher)
                retrieval = {"terms": extract_terms(prompt), "rows": rows, "cypher": safe_cypher}
                st.session_state.graph_last_cypher_model = cypher_model
                st.session_state.graph_cypher_failovers = cypher_failovers
            else:
                retrieval = query_graph(prompt, limit=row_limit)
        else:
            retrieval = {"terms": extract_terms(prompt), "rows": [], "cypher": "Graph retrieval disabled in Notes only mode."}

        graph_context = build_graph_context(retrieval["rows"])
        doc_hits = retrieve_supporting_docs(prompt) if use_notes else []
        document_context = build_document_context(doc_hits)
        graph_help = summarize_graph_contribution(retrieval["rows"])
        st.session_state.graph_last_rows = retrieval["rows"]
        st.session_state.graph_last_cypher = retrieval["cypher"]
        st.session_state.graph_last_context = graph_context
        st.session_state.graph_last_doc_hits = doc_hits
        st.session_state.graph_last_doc_context = document_context
        st.session_state.graph_last_graph_help = graph_help

        messages = build_messages(
            history=st.session_state.graph_chat_history[:-1],
            system_prompt=st.session_state.graph_system_prompt,
            question=prompt,
            graph_context=graph_context,
            document_context=document_context,
            evidence_mode=evidence_mode,
        )
        response, used_model, failovers = invoke_with_model_fallback(
            messages=messages,
            model_chain=model_chain,
            temperature=temperature,
        )
        st.session_state.graph_last_model = used_model
        st.session_state.graph_model_failovers = failovers
        answer = extract_response_text(response, model_name=used_model)
    except Exception as exc:
        st.session_state.graph_last_model = ""
        st.session_state.graph_model_failovers = []
        st.session_state.graph_last_cypher_model = ""
        st.session_state.graph_cypher_failovers = []
        st.session_state.graph_last_doc_hits = []
        st.session_state.graph_last_doc_context = ""
        st.session_state.graph_last_graph_help = ""
        answer = f"Request failed: {exc}"

    st.session_state.graph_chat_history.append(("assistant", answer))
    with st.chat_message("assistant"):
        st.markdown(answer)


if __name__ == "__main__":
    main()
