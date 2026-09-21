"""
core/ollama_client.py

Async client for locally-hosted Ollama models. Mirrors the interface of
call_openrouter() so the orchestrator can dispatch to either provider
transparently.

Ollama exposes an OpenAI-compatible API at http://<host>:11434.
"""

import asyncio
import logging
import os
from typing import Dict, Any, List, Optional, Tuple

import aiohttp

logger = logging.getLogger("ollama_client")

# Default Ollama server endpoint (overridable via OLLAMA_HOST env var)
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"

# Pre-configured ~2B parameter models downloaded via job.sh
OLLAMA_MODELS = [
    {
        "id": "ollama/gemma2:2b",
        "name": "Google Gemma 2 (2B) — Local Ollama",
        "ollama_tag": "gemma2:2b",
        "context_length": 8192,
        "description": "Google DeepMind's efficient 2B instruction-tuned model",
    },
    {
        "id": "ollama/qwen2.5:1.5b",
        "name": "Qwen 2.5 (1.5B) — Local Ollama",
        "ollama_tag": "qwen2.5:1.5b",
        "context_length": 32768,
        "description": "Alibaba's compact multilingual model",
    },
    {
        "id": "ollama/smollm:1.7b",
        "name": "SmolLM (1.7B) — Local Ollama",
        "ollama_tag": "smollm:1.7b",
        "context_length": 8192,
        "description": "HuggingFace's efficiency-focused small language model",
    },
]

# Map orchestrator-facing model IDs to Ollama tags
_ID_TO_TAG = {m["id"]: m["ollama_tag"] for m in OLLAMA_MODELS}


def get_ollama_base_url() -> str:
    """Returns the Ollama API base URL, respecting OLLAMA_HOST env var."""
    return os.environ.get("OLLAMA_HOST", DEFAULT_OLLAMA_BASE_URL).rstrip("/")


def is_ollama_model(model_id: str) -> bool:
    """Check if a model ID refers to a locally-hosted Ollama model."""
    return model_id.startswith("ollama/")


def get_ollama_tag(model_id: str) -> str:
    """Convert an orchestrator model ID (e.g. 'ollama/gemma2:2b') to
    the Ollama tag (e.g. 'gemma2:2b')."""
    if model_id in _ID_TO_TAG:
        return _ID_TO_TAG[model_id]
    # Fallback: strip the 'ollama/' prefix
    return model_id.removeprefix("ollama/")


async def check_ollama_health() -> bool:
    """Verify that the Ollama server is reachable."""
    base_url = get_ollama_base_url()
    try:
        timeout = aiohttp.ClientTimeout(total=5)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(f"{base_url}/api/tags") as resp:
                return resp.status == 200
    except Exception:
        return False


async def list_ollama_models() -> List[Dict[str, Any]]:
    """Fetch the list of models currently available on the Ollama server."""
    base_url = get_ollama_base_url()
    try:
        timeout = aiohttp.ClientTimeout(total=5)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(f"{base_url}/api/tags") as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return data.get("models", [])
    except Exception as e:
        logger.warning(f"Failed to list Ollama models: {e}")
    return []


async def call_ollama(
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.2,
    max_tokens: int = 2500,
    max_retries: int = 2,
    **kwargs,
) -> Tuple[str, Dict[str, Any]]:
    """
    Calls the Ollama chat completions API.

    Uses the same (content, usage_dict) return signature as call_openrouter()
    for seamless interchangeability in the orchestrator.

    Args:
        model: Model ID in 'ollama/<tag>' format or raw Ollama tag.
        system_prompt: System message content.
        user_prompt: User message content.
        temperature: Sampling temperature.
        max_tokens: Maximum tokens in the response.
        max_retries: Number of retry attempts on failure.

    Returns:
        Tuple of (response_content, usage_dict).
    """
    base_url = get_ollama_base_url()
    ollama_tag = get_ollama_tag(model)

    payload = {
        "model": ollama_tag,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
        },
    }

    last_error = None

    for attempt in range(max_retries):
        try:
            # Ollama can be slow on first load; generous timeout
            timeout = aiohttp.ClientTimeout(total=300)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(
                    f"{base_url}/api/chat", json=payload
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        content = data.get("message", {}).get("content", "")

                        # Build a usage dict compatible with OpenRouter format
                        usage = {
                            "prompt_tokens": data.get("prompt_eval_count", 0),
                            "completion_tokens": data.get("eval_count", 0),
                            "total_duration_ns": data.get("total_duration", 0),
                        }

                        return content, usage

                    else:
                        body = await resp.text()
                        last_error = f"Ollama HTTP {resp.status}: {body[:200]}"
                        logger.warning(
                            f"Ollama error on {ollama_tag} (attempt {attempt + 1}): {last_error}"
                        )

        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            last_error = f"Network error: {e}"
            logger.warning(
                f"Ollama network error on {ollama_tag} (attempt {attempt + 1}): {e}"
            )

        # Brief pause before retry
        if attempt < max_retries - 1:
            await asyncio.sleep(1.0 * (attempt + 1))

    raise RuntimeError(
        f"Ollama call failed for '{ollama_tag}' after {max_retries} attempts. "
        f"Last error: {last_error}"
    )
