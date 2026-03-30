DEMO_OVERVIEW = """
This app is designed to demo a Customer 360 workflow on top of Neo4j.

Suggested story:
1. Show Acme Retail as a high-touch account with open revenue and support risk.
2. Show Bright Foods as a lower-risk renewal account.
3. Contrast relationship shape, owners, and campaign influence across the two accounts.
""".strip()


DEMO_PROMPTS = [
    {
        "title": "Acme 360 Summary",
        "prompt": "Show me how Acme Retail is connected to open opportunities, support cases, and key contacts.",
        "why": "Demonstrates the strongest customer-risk narrative in the sample dataset.",
    },
    {
        "title": "Acme Ownership",
        "prompt": "Who owns the open work related to Acme Retail?",
        "why": "Shows operational accountability across opportunities and cases.",
    },
    {
        "title": "Bright Foods Summary",
        "prompt": "Summarize the customer relationship around Bright Foods.",
        "why": "Shows a lower-risk comparison account with renewal and adoption context.",
    },
    {
        "title": "Campaign Influence",
        "prompt": "Which campaign influenced the Acme Retail opportunity?",
        "why": "Shows how campaign relationships contribute to account intelligence.",
    },
]
