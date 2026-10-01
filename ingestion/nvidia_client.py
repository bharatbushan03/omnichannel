"""
Single source of truth for NVIDIA NIM (build.nvidia.com) API configuration.

NVIDIA's endpoint is OpenAI-compatible, so we use the standard `openai`
SDK pointed at NVIDIA's base_url — no special client library needed.

Free tier: sign up at https://build.nvidia.com, open a model (e.g. Nemotron),
click "Get API Key". Set NVIDIA_API_KEY in your environment / .env file.

Two different model families are used here:
  - Chat/instruct model (Nemotron) for classification, entity extraction,
    and the voice agent's reasoning — set via NVIDIA_CHAT_MODEL.
  - Embedding model for vector search — set via NVIDIA_EMBED_MODEL.
    Nemotron itself is NOT an embedding model; NVIDIA hosts separate
    embedding models on the same endpoint (default below).

IMPORTANT: NVIDIA NIM requires `max_tokens` to be explicitly set on every
chat completion call, or the server returns an error. All call sites in
this codebase set it — do not omit it if you add new call sites.
"""
from __future__ import annotations

import os

import openai

NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"

# Free-tier Nemotron instruct model. Swap for any model ID listed on
# https://build.nvidia.com if you want a different Nemotron variant
# (e.g. "nvidia/llama-3.1-nemotron-70b-instruct" or a newer Nemotron 3 release —
# check the catalog for the current recommended default).
DEFAULT_CHAT_MODEL = os.environ.get("NVIDIA_CHAT_MODEL", "nvidia/llama-3.1-nemotron-70b-instruct")

# NVIDIA-hosted embedding model (Nemotron is not an embedding model — this is
# a separate model family on the same endpoint).
DEFAULT_EMBED_MODEL = os.environ.get("NVIDIA_EMBED_MODEL", "nvidia/nv-embedqa-e5-v5")

# nv-embedqa-e5-v5 outputs 1024-dim vectors — this MUST match
# schemas/vector_payload_schema.py's QDRANT_COLLECTION_CONFIG vector size,
# and your Qdrant collection must be created (or recreated) with this size.
DEFAULT_EMBED_DIM = 1024


def get_nvidia_client() -> openai.OpenAI:
    api_key = os.environ.get("NVIDIA_API_KEY")
    if not api_key:
        raise RuntimeError(
            "NVIDIA_API_KEY not set. Get a free key at https://build.nvidia.com "
            "(open a model, click 'Get API Key') and set it in your .env file."
        )
    return openai.OpenAI(base_url=NVIDIA_BASE_URL, api_key=api_key)


def get_nvidia_async_client() -> openai.AsyncOpenAI:
    api_key = os.environ.get("NVIDIA_API_KEY")
    if not api_key:
        raise RuntimeError(
            "NVIDIA_API_KEY not set. Get a free key at https://build.nvidia.com "
            "(open a model, click 'Get API Key') and set it in your .env file."
        )
    return openai.AsyncOpenAI(base_url=NVIDIA_BASE_URL, api_key=api_key)
