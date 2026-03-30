# Customer 360 Graph RAG Chatbot

Customer 360 Graph RAG chatbot built with `Streamlit`, `LangChain`, the Gemini Developer API, and a local `Neo4j` database.

## What This Project Does

- connects to a local Neo4j instance
- retrieves customer relationship evidence for accounts, contacts, opportunities, cases, campaigns, and owners
- turns matching nodes and relationships into textual Customer 360 context
- sends that graph context to Gemini for answer generation
- automatically falls back to another Gemini model when the current one hits quota or rate limits
- shows the last retrieved graph context and Cypher in the UI

## Project Files

- `graph_rag_app.py` - Streamlit entrypoint
- `graph_rag_config.py` - shared config and defaults
- `graph_rag_auth.py` - API key helpers
- `graph_rag_llm.py` - Gemini model selection and fallback logic
- `graph_rag_retrieval.py` - Neo4j retrieval and graph-context formatting
- `graph_rag_state.py` - chat state and message building
- `graph_rag_ui.py` - Streamlit sidebar and diagnostic panels
- `graph_rag_demo.py` - built-in demo prompts and story framing
- `data/customer360_seed.json` - sample Customer 360 graph seed data
- `load_customer360_seed.py` - loader script for local Neo4j
- `DEMO.md` - step-by-step demo script
- `knowledge_base/` - optional narrative account notes for future hybrid RAG use
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

5. Load the sample Customer 360 dataset:

```powershell
python load_customer360_seed.py
```

## Run

```powershell
streamlit run graph_rag_app.py
```

The app includes a `Demo Guide` panel with prebuilt Customer 360 prompts you can load and run directly.

## Customer 360 Use Case

This project is designed for questions like:

- "Show me how Acme is connected to open opportunities, support cases, and key contacts."
- "Summarize the customer relationship around Bright Foods."
- "Which owners, campaigns, and open issues are linked to this account?"

The app works best when the graph contains business entities such as:

- `Account`
- `Contact`
- `Opportunity`
- `Case`
- `Campaign`
- `User`

and relationship paths such as:

- account to contact
- account to opportunity
- account to case
- campaign to opportunity
- user ownership of opportunities or cases

The included seed file creates:

- 2 accounts
- 3 contacts
- 2 opportunities
- 2 cases
- 2 campaigns
- 2 users
- campaign influence and ownership relationships

The included `knowledge_base/` files provide narrative context that matches the seeded accounts and can be used later if you extend the app into a hybrid graph-plus-document RAG workflow.

## Demo Support

You can demo the app in two ways:

- use the built-in `Demo Guide` panel in the UI to load prepared prompts
- follow the scripted walkthrough in [DEMO.md](C:/Users/tiger/project/gemini-streamlit-graph-rag-chatbot/DEMO.md)

## Notes

- This is a sample Customer 360 Graph RAG project, not a production graph planner.
- The Cypher is still generic, but the prompts and output are tuned for account-centered relationship analysis.
- For stronger results, replace the generic Cypher with domain-specific account traversal rules.
- The app only switches models automatically for quota and rate-limit style failures, not for invalid prompts, auth failures, or other hard errors.
