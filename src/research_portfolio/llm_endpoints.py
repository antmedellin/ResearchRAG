"""Single source of truth for model endpoints.

Every model call goes through the LLM gateway (llama-swap). Override any URL
with its own environment variable; otherwise it is built from LLM_GATEWAY_URL.
"""
import os


def _env(name, default=""):
    return (os.getenv(name) or default).strip().rstrip("/")


LLM_GATEWAY_URL = _env("LLM_GATEWAY_URL", "http://llama-swap:8080")

CHAT_URL = _env("CHAT_URL", f"{LLM_GATEWAY_URL}/upstream/gemma4-12b/api/chat")
CHAT_MODEL = _env("CHAT_MODEL", "gemma4:12b")

DENSE_EMBEDDING_URL = _env("DENSE_EMBEDDING_URL", f"{LLM_GATEWAY_URL}/upstream/mxbai-embed/api/embed")
DENSE_EMBEDDING_MODEL = _env("DENSE_EMBEDDING_MODEL", "mxbai-embed-large:335m")

DECISION_URL = _env("DECISION_URL", f"{LLM_GATEWAY_URL}/upstream/clef-flash-9b/v1/systemone")
DECISION_MODEL = _env("DECISION_MODEL", "")

CHAT_API = _env("CHAT_API", "ollama").lower()   # ollama | openai

EMBED_API = _env("EMBED_API", "ollama").lower()   # ollama | openai
