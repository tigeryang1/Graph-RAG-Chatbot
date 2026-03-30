import os

from dotenv import dotenv_values

from graph_rag_config import ENV_PATH


def mask_secret(value: str | None) -> str:
    if not value:
        return "not set"
    if len(value) <= 8:
        return "*" * len(value)
    return f"{value[:4]}...{value[-4:]}"


def get_api_key_status() -> tuple[str, str | None]:
    env_value = os.environ.get("GEMINI_API_KEY")
    if env_value:
        return "environment", env_value
    file_value = dotenv_values(ENV_PATH).get("GEMINI_API_KEY")
    if file_value:
        return ".env file present but not loaded", file_value
    return "not found", None


def get_api_key() -> str:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("Missing GEMINI_API_KEY. Set it in your environment or a .env file.")
    return api_key
