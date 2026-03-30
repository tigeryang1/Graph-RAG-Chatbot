# Gemini Streamlit Graph RAG Chatbot

Sample Graph RAG chatbot built with `Streamlit`, `LangChain`, the Gemini Developer API, and a local `Neo4j` database.

## What This Project Does

- connects to a local Neo4j instance
- searches the graph with a generic Cypher query based on question keywords
- turns matching nodes and relationships into textual graph context
- sends that graph context to Gemini for answer generation
- automatically falls back to another Gemini model when the current one hits quota or rate limits
- shows the last retrieved graph context and Cypher in the UI

## Project Files

- `graph_rag_app.py` - Streamlit UI
- `graph_rag_utils.py` - Neo4j retrieval and Gemini helper functions
- `requirements.txt` - Python dependencies
- `.env.example` - environment template
- `tests/test_graph_rag_utils.py` - small utility tests

## Setup

1. Create and activate a virtual environment.
2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. Copy `.env.example` to `.env` and set your values:

```powershell
Copy-Item .env.example .env
```

Optional model fallback chain:

```text
GEMINI_MODEL=Gemini 2.5 Flash
GEMINI_FALLBACK_MODELS=Gemini 3 Flash,Gemini 2.5 Flash Lite,Gemini 3.1 Flash Lite
GEMINI_AVAILABLE_MODELS=Gemini 2.5 Flash,Gemini 3 Flash,Gemini 2.5 Flash Lite,Gemini 3.1 Flash Lite
```

4. Make sure local Neo4j is running, for example at:

```text
bolt://localhost:7687
```

## Run

```powershell
streamlit run graph_rag_app.py
```

## Expected Neo4j Shape

This sample works best when your graph has descriptive node properties such as:

- `name`
- `title`
- `description`
- `summary`

The generic Cypher query scans node property values for the extracted keywords, then expands one hop to connected nodes.

## Notes

- This is a sample Graph RAG project, not a production graph planner.
- The Cypher is intentionally generic so it can run against many local graphs.
- For stronger results, replace the generic Cypher with domain-specific graph traversal rules.
- The app only switches models automatically for quota and rate-limit style failures, not for invalid prompts, auth failures, or other hard errors.
