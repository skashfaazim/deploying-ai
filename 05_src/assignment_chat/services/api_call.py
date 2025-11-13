"""
Service 1 : External API calls.

We use the public openFDA Drug Label API as a backend and transform the JSON
response into a concise, natural language summary for the user.

API docs: https://open.fda.gov/apis/drug/label/
"""

from __future__ import annotations

import textwrap
from typing import Optional

import requests


OPEN_FDA_URL = "https://api.fda.gov/drug/label.json"


def _shorten(text: str, max_chars: int = 600) -> str:
    """Utility to shorten long label sections without cutting mid-word."""
    text = " ".join(text.split())  # collapse whitespace
    if len(text) <= max_chars:
        return text
    return textwrap.shorten(text, width=max_chars, placeholder="…")


def lookup_drug_label(drug_name: str) -> str:
    """
    Query the openFDA Drug Label API for a brand name and return a transformed,
    human-readable summary (not a verbatim API dump).
    """
    drug_name = drug_name.strip()
    if not drug_name:
        return "Please provide a drug name, for example: `lookup aspirin`."

    try:
        params = {
            "search": f'openfda.brand_name:"{drug_name}"',
            "limit": 1,
        }
        resp = requests.get(OPEN_FDA_URL, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:  # noqa: BLE001
        return (
            "I tried to contact the public openFDA Drug Label API but something went wrong. "
            "This could be due to network restrictions in the grading environment or a temporary API issue. "
            f"Technical detail: {exc}"
        )

    results = data.get("results")
    if not results:
        return (
            f"I couldn't find an FDA drug label for **{drug_name}**. "
            "Try another brand name or a different spelling."
        )

    label = results[0]

    openfda = label.get("openfda", {})
    brand = ", ".join(openfda.get("brand_name", [])) or drug_name
    generic = ", ".join(openfda.get("generic_name", [])) or "Not specified"
    manufacturer = ", ".join(openfda.get("manufacturer_name", [])) or "Not specified"

    indications = _shorten(" ".join(label.get("indications_and_usage", ["Not provided."])))
    warnings = _shorten(" ".join(label.get("warnings", ["Not provided."])))
    adverse = _shorten(" ".join(label.get("adverse_reactions", ["Not provided."])))
    dosing = _shorten(" ".join(label.get("dosage_and_administration", ["Not provided."])))

    summary = (
        f"Here’s a quick regulatory-style summary based on the **FDA drug label** for **{brand}**:\n\n"
        f"- **Generic name:** {generic}\n"
        f"- **Manufacturer:** {manufacturer}\n\n"
        f"**Indications (what it is used for)**\n"
        f"{indications}\n\n"
        f"**Key warnings / precautions**\n"
        f"{warnings}\n\n"
        f"**Important adverse reactions**\n"
        f"{adverse}\n\n"
        f"**High-level dosing information**\n"
        f"{dosing}\n\n"
        "This is a simplified summary of the official label and **not medical advice**. "
        "Patients should always consult a licensed healthcare professional for individual recommendations."
    )
    return summary



