import os
from typing import Any

from graph_rag_auth import get_api_key
from graph_rag_config import DEFAULT_MODEL_OPTIONS, MODEL_NAME_ALIASES

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
except ModuleNotFoundError:  # pragma: no cover - optional until installed
    ChatGoogleGenerativeAI = None


STRUCTURED_CONTENT_MODELS = {
    "Gemini 3.1 Flash Lite",
    "gemini-3.1-flash-lite-preview",
}


def normalize_model_name(model_name: str) -> str:
    return MODEL_NAME_ALIASES.get(model_name, model_name)


def build_llm(model_name: str, temperature: float) -> Any:
    if ChatGoogleGenerativeAI is None:
        raise ModuleNotFoundError("langchain-google-genai is required to run the Graph RAG app.")

    return ChatGoogleGenerativeAI(
        model=normalize_model_name(model_name),
        google_api_key=get_api_key(),
        temperature=temperature,
    )


def uses_structured_content_blocks(model_name: str) -> bool:
    normalized = normalize_model_name(model_name)
    return model_name in STRUCTURED_CONTENT_MODELS or normalized in STRUCTURED_CONTENT_MODELS


def get_available_models() -> list[str]:
    env_value = os.getenv("GEMINI_AVAILABLE_MODELS", "")
    configured = [item.strip() for item in env_value.split(",") if item.strip()]
    models = configured or DEFAULT_MODEL_OPTIONS
    return list(dict.fromkeys(models))


def parse_model_chain(primary_model: str, fallback_models: list[str] | None = None) -> list[str]:
    configured = fallback_models or []
    if not configured:
        env_value = os.getenv("GEMINI_FALLBACK_MODELS", "")
        configured = [item.strip() for item in env_value.split(",") if item.strip()]

    chain: list[str] = []
    for candidate in [primary_model, *configured, *get_available_models()]:
        if candidate and candidate not in chain:
            chain.append(candidate)
    return chain


def is_model_limit_error(exc: Exception) -> bool:
    message = str(exc).lower()
    signals = [
        "429",
        "quota",
        "rate limit",
        "resource exhausted",
        "resource_exhausted",
        "too many requests",
        "exceeded",
        "limit reached",
        "invalid_argument",
        "unexpected model name format",
        "model not found",
        "unsupported model",
    ]
    return any(signal in message for signal in signals)


def invoke_with_model_fallback(
    messages: list[Any],
    model_chain: list[str],
    temperature: float,
    llm_builder=build_llm,
) -> tuple[Any, str, list[str]]:
    if not model_chain:
        raise ValueError("Model chain is empty.")

    errors: list[str] = []
    last_exc: Exception | None = None
    for index, model_name in enumerate(model_chain):
        try:
            llm = llm_builder(model_name=model_name, temperature=temperature)
            response = llm.invoke(messages)
            return response, model_name, errors
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if index == len(model_chain) - 1 or not is_model_limit_error(exc):
                raise
            errors.append(f"{model_name}: {exc}")

    if last_exc is not None:
        raise last_exc
    raise RuntimeError("Model invocation failed without an exception.")


def extract_response_text(response: Any, model_name: str | None = None) -> str:
    content = getattr(response, "content", response)
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                text = item
            elif isinstance(item, dict):
                text = item.get("text", "")
            else:
                text = getattr(item, "text", "")

            if text:
                parts.append(str(text).strip())

        if parts:
            return "\n\n".join(part for part in parts if part)

    return str(content)
