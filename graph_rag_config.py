import os
from pathlib import Path

from dotenv import load_dotenv


APP_DIR = Path(__file__).resolve().parent
ENV_PATH = APP_DIR / ".env"

load_dotenv(dotenv_path=ENV_PATH)

DEFAULT_MODEL_OPTIONS = [
    "Gemini 3.1 Flash Lite",
    "Gemini 3 Flash",
    "Gemini 2.5 Flash",
    "Gemini 2.5 Flash Lite",
]
DEFAULT_RETRIEVAL_MODE = "Fixed query"
DEFAULT_SYSTEM_PROMPT = (
    "You are a Customer 360 Graph RAG assistant. "
    "Use the retrieved Neo4j relationship context to explain how an account is connected to contacts, "
    "opportunities, cases, campaigns, and owners. If the graph evidence is weak or missing, say that clearly."
)
MODEL_NAME_ALIASES = {
    "Gemini 3.1 Flash Lite": "gemini-3.1-flash-lite-preview",
    "Gemini 3 Flash": "gemini-3-flash-preview",
    "Gemini 2.5 Flash": "gemini-2.5-flash",
    "Gemini 2.5 Flash Lite": "gemini-2.5-flash-lite",
}
