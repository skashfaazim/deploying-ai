"""
MedInsight : Assignment 2 project package.

This module:
- Loads secrets from 05_src/.secrets (e.g., OPENAI_API_KEY=...)
- Exposes a shared OpenAI client and config constants used by all services.
"""

from __future__ import annotations

import os
from pathlib import Path
from openai import OpenAI


def load_secrets():
    """
    Load key=value pairs from 05_src/.secrets into environment variables.

    Expected format (one per line, no quotes):
        OPENAI_API_KEY=sk-....
        SOME_OTHER_KEY=value

    Lines starting with # or empty lines are ignored.
    """
    # This file lives in 05_src/assignment_chat/__init__.py
    # So the .secrets file is at 05_src/.secrets (one level up from this folder).
    current_dir = Path(__file__).resolve().parent  # .../05_src/assignment_chat
    secrets_path = current_dir.parent / ".secrets"  # .../05_src/.secrets

    if not secrets_path.exists():
        # Fail silently; user might be using environment variables instead.
        return

    with secrets_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip()
            # Do not overwrite variables that are already set in the environment
            if key and key not in os.environ:
                os.environ[key] = value


# Load secrets before creating the OpenAI client
load_secrets()

# Shared OpenAI client
client = OpenAI()

# Main chat model (small but capable)
MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# Embedding model for Chroma
EMBED_MODEL_NAME = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
