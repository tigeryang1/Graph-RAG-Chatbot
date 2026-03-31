# Customer 360 Graph RAG Chatbot

Graph-first Customer 360 demo built with `Streamlit`, `LangChain`, Gemini, and a local `Neo4j` database.

The app is designed to answer account questions such as:

- Which contacts, opportunities, cases, and campaigns are connected to this customer?
- Who owns the open work for this account?
- What changes when I answer from graph data only, note files only, or both?

## What The App Does

- queries a local Neo4j graph that contains Salesforce-like customer relationship data
- optionally retrieves matching note files from `knowledge_base/`
- sends the selected evidence to Gemini for answer generation
- supports safe dynamic Cypher as an opt-in retrieval mode
- shows the evidence used for the latest answer:
  - graph rows
  - Cypher query
  - matched note files
  - graph contribution summary

## Evidence Modes

The app supports three evidence modes:

- `Graph + Notes`
  - use Neo4j relationship evidence and matching note files together
- `Graph only`
  - use only Neo4j graph retrieval
- `Notes only`
  - skip Neo4j and answer only from the local note files

This makes it easy to demo how graph retrieval changes the answer quality.

## Retrieval Modes

The app supports two graph retrieval modes:

- `Fixed query`
  - deterministic account-centered Cypher
  - good default for demos and predictable behavior
- `Dynamic Cypher (safe)`
  - Gemini proposes Cypher
  - the app validates it locally before execution
  - only read-only queries against the allowed Customer 360 schema are permitted

## Customer 360 Data Model

The sample graph uses these main node types:

- `Account`
- `Contact`
- `Opportunity`
- `Case`
- `Campaign`
- `User`

And these main relationship types:

- `WORKS_FOR`
- `FOR_ACCOUNT`
- `OWNS`
- `TARGETS`
- `INFLUENCED`

The seeded data includes:

- 2 accounts
- 3 contacts
- 2 opportunities
- 2 cases
- 2 campaigns
- 2 users

The sample dataset is intentionally small so the demo stays readable.

## Project Layout

- `graph_rag_app.py`
  - Streamlit entrypoint
- `graph_rag_config.py`
  - defaults and environment bootstrap
- `graph_rag_auth.py`
  - Gemini API key helpers
- `graph_rag_llm.py`
  - Gemini model mapping, fallback, and response text extraction
- `graph_rag_retrieval.py`
  - Neo4j queries, safe dynamic Cypher validation, local note retrieval
- `graph_rag_state.py`
  - chat state and prompt assembly
- `graph_rag_ui.py`
  - sidebar, demo panel, evidence panels, diagnostics
- `graph_rag_demo.py`
  - built-in demo prompts
- `data/customer360_seed.json`
  - sample graph seed data
- `load_customer360_seed.py`
  - loader for local Neo4j
- `knowledge_base/`
  - local narrative account notes used in note-based retrieval
- `DEMO.md`
  - demo script

## Local Setup

1. Create and activate a virtual environment.

2. Install dependencies.

```powershell
pip install -r requirements.txt
```

3. Copy `.env.example` to `.env`.

```powershell
Copy-Item .env.example .env
```

4. Set at least:

```text
GEMINI_API_KEY=your_gemini_api_key
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_neo4j_password
```

Optional Gemini model config:

```text
GEMINI_MODEL=Gemini 3.1 Flash Lite
GEMINI_FALLBACK_MODELS=Gemini 3 Flash,Gemini 2.5 Flash,Gemini 2.5 Flash Lite
GEMINI_AVAILABLE_MODELS=Gemini 3.1 Flash Lite,Gemini 3 Flash,Gemini 2.5 Flash,Gemini 2.5 Flash Lite
```

5. Start your local Neo4j DBMS.

6. Load the sample Customer 360 graph.

```powershell
python load_customer360_seed.py
```

## Run The App

```powershell
streamlit run graph_rag_app.py
```

## Demo Flow

The easiest demo path is:

1. Start in `Graph + Notes`
2. Use `Fixed query`
3. Run one of the built-in demo prompts from the `Demo Guide`

Suggested prompts:

- `Show me how Acme Retail is connected to open opportunities, support cases, and key contacts.`
- `Who owns the open work related to Acme Retail?`
- `Summarize the customer relationship around Bright Foods.`
- `Which campaign influenced the Acme Retail opportunity?`

Then switch evidence modes:

- `Graph + Notes`
- `Graph only`
- `Notes only`

That contrast makes the value of graph retrieval obvious.

## What The UI Shows

For the latest answer, the app shows:

- evidence mode
- retrieval mode
- graph rows retrieved
- files used from `knowledge_base/`
- how the graph contributed to the answer
- the last Cypher query used
- Gemini model and fallback diagnostics

## Notes

- This is a demo project, not a production graph planner.
- The fixed query path is the most reliable mode for account-centered demos.
- Dynamic Cypher is intentionally constrained to a small allowed schema and read-only operations.
- The local note retrieval is lightweight and file-based, not a vector database.
- Gemini fallback handles quota and rate-limit style failures, and also helps when a selected model name is rejected.

