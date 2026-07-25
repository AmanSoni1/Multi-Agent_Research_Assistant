"""Shared helpers used by every agent node: timed LLM calls, robust JSON parsing, token tracking."""
from __future__ import annotations
import json
import logging
import re
import time
from typing import Any, Dict, List, Tuple

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

logger = logging.getLogger(__name__)

MAX_RETRIES = 2
RETRY_BACKOFF_SECONDS = 1.5


def _strip_code_fences(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()
    return text


def call_llm_json(
    llm: BaseChatModel,
    system_prompt: str,
    user_prompt: str,
    node_name: str,
) -> Tuple[Dict[str, Any], float, Dict[str, int]]:
    """
    Call an LLM expecting a JSON object back. Retries on transient failures
    (timeouts, malformed JSON) with backoff. Returns (parsed_json, seconds_elapsed, token_usage).

    Raises RuntimeError only after all retries are exhausted -- callers should
    decide how to degrade (e.g. skip a sub-question, mark evidence as empty)
    rather than crash the whole graph run.
    """
    last_error: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 2):
        start = time.time()
        try:
            response = llm.invoke(
                [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
            )
            elapsed = time.time() - start
            raw = response.content if isinstance(response.content, str) else str(response.content)
            cleaned = _strip_code_fences(raw)
            parsed = json.loads(cleaned)

            usage = getattr(response, "usage_metadata", None) or {}
            tokens = {
                "input_tokens": usage.get("input_tokens", 0),
                "output_tokens": usage.get("output_tokens", 0),
            }
            return parsed, elapsed, tokens

        except json.JSONDecodeError as exc:
            last_error = exc
            logger.warning(
                "[%s] attempt %d/%d: model returned non-JSON, retrying: %s",
                node_name, attempt, MAX_RETRIES + 1, str(exc)[:200],
            )
        except Exception as exc:  # noqa: BLE001 -- network/timeout/rate-limit errors, all retryable
            last_error = exc
            logger.warning(
                "[%s] attempt %d/%d: LLM call failed, retrying: %s",
                node_name, attempt, MAX_RETRIES + 1, exc,
            )
        time.sleep(RETRY_BACKOFF_SECONDS * attempt)

    raise RuntimeError(f"[{node_name}] LLM call failed after {MAX_RETRIES + 1} attempts: {last_error}")


def dedupe_evidence(evidence: List[dict]) -> List[dict]:
    """Drop exact-duplicate evidence chunks (same source_id + sub_question_id)."""
    seen = set()
    out = []
    for chunk in evidence:
        key = (chunk["sub_question_id"], chunk["source_id"])
        if key not in seen:
            seen.add(key)
            out.append(chunk)
    return out
