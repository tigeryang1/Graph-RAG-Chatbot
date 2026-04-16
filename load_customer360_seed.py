from __future__ import annotations

import json
from pathlib import Path

import graph_rag_config
from graph_retrieval import close_driver, get_driver, get_neo4j_settings

try:
    from neo4j import GraphDatabase
except ModuleNotFoundError as exc:  # pragma: no cover - runtime dependency
    raise ModuleNotFoundError("neo4j is required to load the Customer 360 seed data.") from exc


APP_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_FILE = APP_DIR / "data" / "customer360_seed.json"


def load_seed_file(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def mask_secret(value: str | None) -> str:
    if not value:
        return "not set"
    if len(value) <= 8:
        return "*" * len(value)
    return f"{value[:2]}...{value[-2:]}"


def run_write(tx, query: str, rows: list[dict]) -> None:
    if not rows:
        return
    tx.run(query, rows=rows)


def count_rows(data: dict) -> dict[str, int]:
    return {key: len(value) for key, value in data.items() if isinstance(value, list)}


def print_diagnostics(settings: dict[str, str | None], data: dict) -> None:
    print("Customer 360 seed loader")
    print(f"- Data file: {DEFAULT_DATA_FILE}")
    print(f"- Neo4j URI: {settings['uri']}")
    print(f"- Neo4j username: {settings['username']}")
    print(f"- Neo4j password: {mask_secret(settings['password'])}")
    for key, value in count_rows(data).items():
        print(f"- {key}: {value}")


def main() -> None:
    settings = get_neo4j_settings()
    if not settings["password"]:
        raise ValueError("Missing NEO4J_PASSWORD. Set it in your environment or a .env file.")

    data = load_seed_file(DEFAULT_DATA_FILE)
    print_diagnostics(settings, data)
    print("Connecting to Neo4j...")
    driver = get_driver()

    account_query = """
    UNWIND $rows AS row
    MERGE (a:Account {id: row.id})
    SET a.name = row.name,
        a.industry = row.industry,
        a.region = row.region,
        a.tier = row.tier,
        a.status = row.status
    """

    contact_query = """
    UNWIND $rows AS row
    MERGE (c:Contact {id: row.id})
    SET c.name = row.name,
        c.email = row.email,
        c.title = row.title
    WITH c, row
    MATCH (a:Account {id: row.account_id})
    MERGE (c)-[:WORKS_FOR]->(a)
    """

    user_query = """
    UNWIND $rows AS row
    MERGE (u:User {id: row.id})
    SET u.name = row.name,
        u.role = row.role
    """

    opportunity_query = """
    UNWIND $rows AS row
    MERGE (o:Opportunity {id: row.id})
    SET o.name = row.name,
        o.stage = row.stage,
        o.amount = row.amount,
        o.close_date = row.close_date,
        o.status = row.status
    WITH o, row
    MATCH (a:Account {id: row.account_id})
    MERGE (o)-[:FOR_ACCOUNT]->(a)
    WITH o, row
    MATCH (u:User {id: row.owner_user_id})
    MERGE (u)-[:OWNS]->(o)
    """

    case_query = """
    UNWIND $rows AS row
    MERGE (c:Case {id: row.id})
    SET c.subject = row.subject,
        c.priority = row.priority,
        c.status = row.status
    WITH c, row
    MATCH (a:Account {id: row.account_id})
    MERGE (c)-[:FOR_ACCOUNT]->(a)
    WITH c, row
    MATCH (u:User {id: row.owner_user_id})
    MERGE (u)-[:OWNS]->(c)
    """

    campaign_query = """
    UNWIND $rows AS row
    MERGE (c:Campaign {id: row.id})
    SET c.name = row.name,
        c.status = row.status,
        c.budget = row.budget
    WITH c, row
    MATCH (a:Account {id: row.account_id})
    MERGE (c)-[:TARGETS]->(a)
    """

    campaign_influence_query = """
    UNWIND $rows AS row
    MATCH (c:Campaign {id: row.campaign_id})
    MATCH (o:Opportunity {id: row.opportunity_id})
    MERGE (c)-[:INFLUENCED]->(o)
    """

    with driver.session() as session:
        print("Connection established. Writing seed data...")
        session.execute_write(run_write, account_query, data.get("accounts", []))
        print(f"- Loaded accounts: {len(data.get('accounts', []))}")
        session.execute_write(run_write, contact_query, data.get("contacts", []))
        print(f"- Loaded contacts: {len(data.get('contacts', []))}")
        session.execute_write(run_write, user_query, data.get("users", []))
        print(f"- Loaded users: {len(data.get('users', []))}")
        session.execute_write(run_write, opportunity_query, data.get("opportunities", []))
        print(f"- Loaded opportunities: {len(data.get('opportunities', []))}")
        session.execute_write(run_write, case_query, data.get("cases", []))
        print(f"- Loaded cases: {len(data.get('cases', []))}")
        session.execute_write(run_write, campaign_query, data.get("campaigns", []))
        print(f"- Loaded campaigns: {len(data.get('campaigns', []))}")
        session.execute_write(run_write, campaign_influence_query, data.get("campaign_influence", []))
        print(f"- Loaded campaign influence links: {len(data.get('campaign_influence', []))}")

    close_driver()
    print(f"Done. Loaded Customer 360 seed data from {DEFAULT_DATA_FILE}")


if __name__ == "__main__":
    main()
