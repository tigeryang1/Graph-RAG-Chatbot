# Root-level conftest.py ensures pytest adds the project directory to sys.path,
# allowing flat-layout modules (graph_rag_llm, graph_retrieval, etc.) to be
# imported from tests without installing the project as a package.
