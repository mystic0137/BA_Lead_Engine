import logging
import time
from typing import Literal

from groq import Groq
from openai import OpenAI

from src.config import GROQ_MODEL, get_settings
from src.rag.prompts import MAX_TOKENS

logger = logging.getLogger(__name__)


Provider = Literal["groq", "together", "ollama"]


_PROVIDER_ORDER: list[Provider] = ["groq", "together", "ollama"]


def _groq_client() -> Groq:
    api_key = get_settings().GROQ_API_KEY.get_secret_value()
    if not api_key:
        raise RuntimeError("GROQ_API_KEY not configured")
    return Groq(api_key=api_key)


def _together_client() -> OpenAI:
    api_key = get_settings().TOGETHER_API_KEY.get_secret_value()
    if not api_key:
        raise RuntimeError("TOGETHER_API_KEY not configured")
    return OpenAI(api_key=api_key, base_url="https://api.together.xyz/v1")


def _ollama_client() -> OpenAI:
    return OpenAI(
        api_key="ollama",
        base_url=f"{get_settings().OLLAMA_BASE_URL}/v1",
    )


_CLIENT_BUILDERS = {
    "groq": _groq_client,
    "together": _together_client,
    "ollama": _ollama_client,
}


def get_providers() -> list[Provider]:
    if get_settings().LLM_FALLBACK_ENABLED:
        return _PROVIDER_ORDER
    return ["groq"]


def _model_for_provider(provider: Provider) -> str:
    match provider:
        case "groq":
            return GROQ_MODEL
        case "together":
            return "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"
        case "ollama":
            return "llama3.1:8b"


def generate_chat(
    messages: list[dict],
    max_tokens: int = MAX_TOKENS,
    temperature: float = 0.6,
) -> dict:
    last_error: Exception | None = None
    for provider in get_providers():
        try:
            client = _CLIENT_BUILDERS[provider]()
            model = _model_for_provider(provider)
            t0 = time.time()
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            latency_ms = round((time.time() - t0) * 1000)
            raw = response.choices[0].message.content.strip()
            return {
                "raw": raw,
                "provider": provider,
                "model": model,
                "tokens_input": response.usage.prompt_tokens,
                "tokens_output": response.usage.completion_tokens,
                "latency_ms": latency_ms,
            }
        except Exception as e:
            logger.warning(
                "LLM provider %s failed: %s. %s",
                provider, e,
                "Trying next provider..." if provider != get_providers()[-1] else "No more providers.",
            )
            last_error = e
            continue
    raise RuntimeError("All LLM providers failed") from last_error
