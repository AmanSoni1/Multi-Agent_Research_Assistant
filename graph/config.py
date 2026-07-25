"""
Central place for model routing decisions. Every agent node imports its LLM
client from here rather than constructing one inline -- this is the "model
routing" story for the interview: one file makes every node's model choice
legible and swappable.
"""
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")

# --- Model routing table -------------------------------------------------
# Change these constants (or the env vars below) to re-route any node.
WRITER_MODEL = "deepseek/deepseek-v4-flash"
CRITIC_MODEL = "deepseek/deepseek-v4-flash"
RERANK_MODEL = "deepseek/deepseek-v4-flash"

# Planner has a pluggable backend: local (Ollama/Gemma) or api (OpenRouter).
# Defaults to "api" so the repo runs for anyone who clones it with just an
# OpenRouter key -- no GPU / Ollama dependency required to demo the system.
PLANNER_BACKEND = os.getenv("PLANNER_BACKEND", "api").lower()  # "local" | "api"
PLANNER_API_MODEL = "deepseek/deepseek-v4-flash"
PLANNER_LOCAL_MODEL = os.getenv("PLANNER_LOCAL_MODEL", "gemma3:4b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

MAX_REVISIONS_DEFAULT = 2


def get_openrouter_llm(model: str, temperature: float = 0.3) -> ChatOpenAI:
    """Any OpenRouter-hosted chat model, called via the OpenAI-compatible endpoint."""
    if not OPENROUTER_API_KEY:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set. Add it to your .env file. "
            "Get a free key at https://openrouter.ai/keys"
        )
    return ChatOpenAI(
        model=model,
        api_key=OPENROUTER_API_KEY,
        base_url=OPENROUTER_BASE_URL,
        temperature=temperature,
    )


def get_planner_llm(temperature: float = 0.4) -> ChatOpenAI:
    """
    Pluggable Planner backend.
    - PLANNER_BACKEND=local  -> Gemma 3 4B via Ollama (OpenAI-compatible endpoint)
    - PLANNER_BACKEND=api    -> Deepseek V4 Flash via OpenRouter (default)
    """
    if PLANNER_BACKEND == "local":
        # Ollama exposes an OpenAI-compatible /v1 endpoint, so we reuse ChatOpenAI.
        return ChatOpenAI(
            model=PLANNER_LOCAL_MODEL,
            api_key="ollama",  # unused, but required by the client
            base_url=f"{OLLAMA_BASE_URL}/v1",
            temperature=temperature,
        )
    return get_openrouter_llm(PLANNER_API_MODEL, temperature=temperature)
