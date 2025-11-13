"""
Guardrails to block restricted topics (exact words only)
"""

import re


# Restricted topics (exact matches, not substrings)
FORBIDDEN = [
    r"\bcat\b",
    r"\bcats\b",
    r"\bdog\b",
    r"\bdogs\b",
    r"\bhoroscope\b",
    r"\bzodiac\b",
    r"\btaylor swift\b",
]

SYSTEM_PROMPT_KEYWORDS = [
    "system prompt",
    "reveal system prompt",
    "change system prompt",
    "modify system prompt",
]


def check_guardrails(user_input: str):
    """
    Returns (allowed: bool, message_if_blocked: Optional[str])
    """

    text = user_input.lower()

    # Guard against system prompt access
    for kw in SYSTEM_PROMPT_KEYWORDS:
        if kw in text:
            return False, "I’m not allowed to reveal or modify my system prompt."

    # Check forbidden topics using whole-word regex (no substring bugs!)
    for pattern in FORBIDDEN:
        if re.search(pattern, text):
            return False, (
                "I'm designed for medication and regulatory questions only. "
                "I cannot discuss cats, dogs, horoscopes, zodiac signs, or Taylor Swift."
            )

    return True, None
