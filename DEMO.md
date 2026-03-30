# Customer 360 Demo Script

## Goal

Demonstrate that the app can explain customer relationships from a Neo4j graph instead of just listing records.

## Setup

1. Start local Neo4j
2. Load the seed data:

```powershell
python load_customer360_seed.py
```

3. Run the app:

```powershell
streamlit run graph_rag_app.py
```

## Demo Flow

### 1. Introduce the scenario

Explain that the graph contains:
- accounts
- contacts
- opportunities
- cases
- campaigns
- users who own work

### 2. Run the Acme summary

Prompt:

```text
Show me how Acme Retail is connected to open opportunities, support cases, and key contacts.
```

Point out:
- open opportunity
- open high-priority case
- key contacts
- relationship path in the retrieved graph context

### 3. Run the ownership question

Prompt:

```text
Who owns the open work related to Acme Retail?
```

Point out:
- opportunity owner
- case owner
- how the graph reveals accountability, not just entities

### 4. Compare with Bright Foods

Prompt:

```text
Summarize the customer relationship around Bright Foods.
```

Point out:
- still active revenue motion
- lower support severity
- contrast with Acme’s higher-risk posture

### 5. Show campaign influence

Prompt:

```text
Which campaign influenced the Acme Retail opportunity?
```

Point out:
- campaign-to-opportunity relationship
- why Graph RAG is useful for connected account intelligence

## Key Message

This is a Customer 360 assistant:
- graph gives structure and connected evidence
- Gemini turns that into a readable business summary
