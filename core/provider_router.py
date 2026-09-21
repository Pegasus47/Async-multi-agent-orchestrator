"""
core/provider_router.py

Unified dispatch layer that routes model calls to the correct backend
(OpenRouter API or local Ollama) based on the model ID prefix.

    - "ollama/*"  →  core.ollama_client.call_ollama()
    - everything else  →  core.openrouter_client.call_openrouter()

This module also provides a combined model listing endpoint that merges
models from both providers.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple

from core.openrouter_client import call_openrouter, fetch_free_models
from core.ollama_client import (
    call_ollama,
    is_ollama_model,
    OLLAMA_MODELS,
    check_ollama_health,
    list_ollama_models,
)

logger = logging.getLogger("provider_router")


async def call_model(
    model: str,
    system_prompt: str,
    user_prompt: str,
    api_key: Optional[str] = None,
    temperature: float = 0.2,
    max_tokens: int = 2500,
    max_retries: int = 3,
) -> Tuple[str, Dict[str, Any]]:
    """
    Route a chat completion request to the appropriate backend.

    Models prefixed with 'ollama/' are sent to the local Ollama server.
    All other models are sent to the OpenRouter API.

    Returns the same (content, usage_dict) tuple regardless of provider.
    """
    if is_ollama_model(model):
        logger.info(f"Routing '{model}' → Ollama (local)")
        return await call_ollama(
            model=model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            max_retries=max_retries,
        )
    else:
        logger.info(f"Routing '{model}' → OpenRouter (API)")
        return await call_openrouter(
            model=model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            max_retries=max_retries,
        )


async def fetch_all_models(api_key: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Return a merged list of available models from both providers.

    OpenRouter models are fetched dynamically (with cache).
    Ollama models are included only if the Ollama server is reachable.
    """
    # Always fetch OpenRouter models
    openrouter_models = await fetch_free_models(api_key)

    # Check if Ollama is available and include local models
    ollama_available = await check_ollama_health()

    if ollama_available:
        # Cross-reference configured models with what's actually pulled
        live_models = await list_ollama_models()
        live_names = {m.get("name", "").split(":")[0] for m in live_models}

        ollama_entries = []
        for m in OLLAMA_MODELS:
            tag_base = m["ollama_tag"].split(":")[0]
            status = "available" if tag_base in live_names else "not_pulled"
            entry = {**m, "provider": "ollama", "status": status}
            ollama_entries.append(entry)

        logger.info(
            f"Ollama server online — {len(ollama_entries)} models configured, "
            f"{sum(1 for e in ollama_entries if e['status'] == 'available')} available"
        )
    else:
        ollama_entries = [
            {**m, "provider": "ollama", "status": "server_offline"}
            for m in OLLAMA_MODELS
        ]
        logger.info("Ollama server offline — models listed but marked unavailable")

    # Tag OpenRouter models with provider info
    for m in openrouter_models:
        m["provider"] = "openrouter"
        m["status"] = "available"

    return ollama_entries + openrouter_models
