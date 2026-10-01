import asyncio
import json
import logging
import os
import random
import time
from typing import Dict, Any, List, Optional, Tuple
import aiohttp

logger = logging.getLogger("openrouter_client")

OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"

# Models named "ollama/<name>" go to a local Ollama server instead of OpenRouter.
OLLAMA_PREFIX = "ollama/"
OLLAMA_DEFAULT_URL = "http://127.0.0.1:11434"

# Fallback curated free models if discovery is offline
DEFAULT_FREE_MODELS = [
    {
        "id": "google/gemma-4-31b-it:free",
        "name": "Google: Gemma 4 31B (Free)",
        "context_length": 32768,
        "description": "High performance instruction-tuned model by Google",
    },
    {
        "id": "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
        "name": "NVIDIA: Nemotron 3 Nano 30B Reasoning (Free)",
        "context_length": 32768,
        "description": "Optimized reasoning model with low latency",
    },
    {
        "id": "cohere/north-mini-code:free",
        "name": "Cohere: North Mini Code (Free)",
        "context_length": 32768,
        "description": "Specialized code generation model by Cohere",
    },
    {
        "id": "nvidia/nemotron-3-super-120b-a12b:free",
        "name": "NVIDIA: Nemotron 3 Super 120B (Free)",
        "context_length": 32768,
        "description": "High-capacity multi-task reasoning and coding model",
    },
    {
        "id": "openrouter/free",
        "name": "OpenRouter: Free Meta Router",
        "context_length": 16384,
        "description": "Routes dynamically across available free models",
    },
]

# In-memory cache for discovered models
_CACHED_FREE_MODELS: Optional[List[Dict[str, Any]]] = None
_CACHE_TIMESTAMP: float = 0
CACHE_TTL_SECONDS = 600  # 10 minutes

# Rate limiter semaphore for concurrent requests to avoid 429
_GLOBAL_SEMAPHORE = asyncio.Semaphore(3)


async def fetch_free_models(api_key: Optional[str] = None) -> List[Dict[str, Any]]:
    """Dynamically fetches active free models from OpenRouter."""
    global _CACHED_FREE_MODELS, _CACHE_TIMESTAMP
    now = time.time()
    if _CACHED_FREE_MODELS and (now - _CACHE_TIMESTAMP < CACHE_TTL_SECONDS):
        return _CACHED_FREE_MODELS

    headers = {"User-Agent": "AsyncMultiAgentOrchestrator/1.0"}
    key = api_key or os.environ.get("OPENROUTER_API_KEY")
    if key:
        headers["Authorization"] = f"Bearer {key}"

    try:
        timeout = aiohttp.ClientTimeout(total=8)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(OPENROUTER_MODELS_URL, headers=headers) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    models = data.get("data", [])
                    free_list = []
                    for m in models:
                        mid = m.get("id", "")
                        pricing = m.get("pricing", {})
                        p_in = pricing.get("prompt", 0)
                        p_out = pricing.get("completion", 0)
                        is_free = (
                            ":free" in mid
                            or (str(p_in) in ["0", "0.0", "0.00"] and str(p_out) in ["0", "0.0", "0.00"])
                        )
                        if is_free:
                            free_list.append({
                                "id": mid,
                                "name": m.get("name", mid),
                                "context_length": m.get("context_length", 8192),
                                "description": m.get("description", "OpenRouter Free Tier Model"),
                                "pricing": pricing,
                            })
                    if free_list:
                        # Sort by name
                        free_list.sort(key=lambda x: x["name"])
                        _CACHED_FREE_MODELS = free_list
                        _CACHE_TIMESTAMP = now
                        return free_list
    except Exception as e:
        logger.warning(f"Failed to fetch dynamic free models ({e}). Using default free models.")

    return DEFAULT_FREE_MODELS


def _resolve_provider(model: str, api_key: Optional[str]) -> Tuple[str, Dict[str, str], str, int]:
    """Return (url, headers, model name to send, timeout in seconds) for a model ID."""
    if model.startswith(OLLAMA_PREFIX):
        # No key needed. Long timeout: the first request after `ollama serve` loads the model.
        base = os.environ.get("OLLAMA_BASE_URL", OLLAMA_DEFAULT_URL)
        url = base.rstrip("/") + "/v1/chat/completions"
        return url, {"Content-Type": "application/json"}, model[len(OLLAMA_PREFIX):], 600

    key = api_key or os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise ValueError("OPENROUTER_API_KEY is not set or provided")
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/Pegasus47/Async-multi-agent-orchestrator",
        "X-Title": "Async Multi-Agent Orchestrator POC",
    }
    return OPENROUTER_API_URL, headers, model, 90


async def call_openrouter(
    model: str,
    system_prompt: str,
    user_prompt: str,
    api_key: Optional[str] = None,
    temperature: float = 0.2,
    max_tokens: int = 2500,
    max_retries: int = 3,
) -> Tuple[str, Dict[str, Any]]:
    """
    Calls OpenRouter chat completions with semaphore rate-limiting and exponential backoff retry.
    Returns (content, usage_dict).
    """
    url, headers, model, timeout_s = _resolve_provider(model, api_key)

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    last_error = None

    for attempt in range(max_retries):
        try:
            async with _GLOBAL_SEMAPHORE:
                timeout = aiohttp.ClientTimeout(total=timeout_s)
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    async with session.post(url, headers=headers, json=payload) as resp:
                        status = resp.status
                        data = await resp.json()

                        if status == 200:
                            if "choices" in data and len(data["choices"]) > 0:
                                content = data["choices"][0]["message"]["content"]
                                usage = data.get("usage", {})
                                return content, usage
                            elif "error" in data:
                                raise RuntimeError(f"OpenRouter error: {data['error']}")
                            else:
                                raise RuntimeError(f"Unexpected response format: {data}")

                        elif status in [429, 502, 503, 504]:
                            # Rate limit or temporary service issue -> back off
                            wait_s = (2 ** attempt) + random.uniform(0.5, 1.5)
                            logger.warning(f"Rate limited or status {status} on {model}. Retrying in {wait_s:.1f}s...")
                            await asyncio.sleep(wait_s)
                            last_error = f"HTTP {status}: {data.get('error', 'Rate limited or gateway error')}"
                            continue
                        else:
                            err_msg = data.get("error", f"HTTP {status}")
                            raise RuntimeError(f"OpenRouter API error ({status}): {err_msg}")

        except (aiohttp.ClientError, asyncio.TimeoutError) as net_err:
            wait_s = (2 ** attempt) + random.uniform(0.5, 1.5)
            logger.warning(f"Network error on {model} attempt {attempt+1}: {net_err}. Retrying in {wait_s:.1f}s...")
            await asyncio.sleep(wait_s)
            last_error = str(net_err)
        except Exception as ex:
            if attempt == max_retries - 1:
                raise ex
            wait_s = (2 ** attempt) + 0.5
            await asyncio.sleep(wait_s)
            last_error = str(ex)

    raise RuntimeError(f"Failed after {max_retries} attempts. Last error: {last_error}")
