"""
Service 2 : Semantic search over Health Canada Notice of Compliance (NOC) data.

Features:
- Auto-downloads the NOC Drug Product JSON from Health Canada if the local
  file is missing or invalid.
- Cleans and chunks large NOC entries to stay within OpenAI embedding token
  limits.
- Uses a custom embedding adapter compatible with openai>=1.0.0.
- Stores embeddings in a persistent ChromaDB collection.
"""

from __future__ import annotations

import json
import urllib.request
import urllib.error
from pathlib import Path
from typing import List, Dict, Tuple

import chromadb
from chromadb.api.types import Documents, Embeddings

from .. import client, MODEL_NAME, EMBED_MODEL_NAME


# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------

CURRENT_DIR = Path(__file__).resolve().parent
DATA_DIR = CURRENT_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

DATA_PATH = DATA_DIR / "noc_drug_products.json"
CHROMA_PATH = CURRENT_DIR.parent / "chroma_store"
COLLECTION_NAME = "noc_drug_products"

# Official NOC Drug Product API endpoint (English JSON)
# If Health Canada ever changes this, only this constant needs updating.
NOC_DRUG_URL = (
    "https://health-products.canada.ca/noc/api/drugproduct/?lang=en&type=json"
)

# Character limit per chunk when preparing docs for embeddings
CHUNK_SIZE = 2000  # ~1.5k tokens max; very safe for embeddings
BATCH_SIZE = 128   # Number of chunks per Chroma add() call

# Performance tweak: only index a subset of the NOC dataset
# This keeps build time reasonable for the assignment while still
# demonstrating semantic search clearly.
MAX_ENTRIES = 1500  # adjust down (e.g. 800) if you want even faster indexing


# ---------------------------------------------------------------------------
# Auto-download & robust loading
# ---------------------------------------------------------------------------


def _download_noc_file() -> None:
    """
    Download the NOC Drug Product JSON from Health Canada and save it
    to DATA_PATH. If download fails, an informative exception is raised.
    """
    try:
        print(f"[semantic] Downloading NOC data from {NOC_DRUG_URL} ...")
        with urllib.request.urlopen(NOC_DRUG_URL) as resp:
            raw = resp.read().decode("utf-8")
        DATA_PATH.write_text(raw, encoding="utf-8")
        print("[semantic] Download complete and saved to data/noc_drug_products.json")
    except urllib.error.URLError as e:
        raise RuntimeError(
            f"Failed to download NOC data from {NOC_DRUG_URL}: {e}"
        ) from e


def _ensure_noc_file() -> None:
    """
    Ensure that the local NOC JSON file exists and contains valid JSON.
    If it doesn't exist or is invalid, attempt to download a fresh copy.
    """
    if not DATA_PATH.exists():
        _download_noc_file()
        return

    # Try to parse existing file; if invalid, re-download.
    try:
        raw = DATA_PATH.read_text(encoding="utf-8")
        json.loads(raw)  # just to validate
    except Exception:
        print("[semantic] Existing noc_drug_products.json is invalid; re-downloading.")
        _download_noc_file()


def _load_noc_data() -> List[Dict]:
    """
    Load the NOC dataset from noc_drug_products.json.

    - Ensures the file exists and is valid (auto-downloads if needed).
    - Parses as JSON; expects a list of dicts.
    """
    _ensure_noc_file()
    raw = DATA_PATH.read_text(encoding="utf-8").strip()

    data = json.loads(raw)
    if isinstance(data, dict):
        data = [data]
    if not isinstance(data, list):
        raise RuntimeError(
            "Unexpected JSON structure for NOC data: expected a list of objects."
        )
    return data  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Text cleaning & chunking
# ---------------------------------------------------------------------------


def _entry_to_base_text(entry: Dict) -> str:
    """
    Convert a NOC entry to a readable text block highlighting key fields.
    This keeps semantic search focused on important regulatory information.
    """
    brand = entry.get("noc_product_name") or entry.get("brand_name") or "Unknown brand"
    manufacturer = entry.get("noc_manufacturer_name") or entry.get("manufacturer") or "Unknown manufacturer"
    therapeutic_class = entry.get("noc_therapeutic_class") or entry.get("therapeutic_class") or ""
    indication = entry.get("noc_reason_submission") or entry.get("submission_reason") or ""
    submission_class = entry.get("noc_submission_class") or entry.get("submission_class") or ""
    product_type = entry.get("noc_product_type") or entry.get("product_type") or ""
    noc_date = entry.get("noc_date") or ""
    last_update = entry.get("noc_last_update_date") or ""
    country = entry.get("noc_crp_country_name") or ""
    number = entry.get("noc_number") or ""

    parts = [
        f"NOC number: {number}",
        f"Brand / Product name: {brand}",
        f"Manufacturer: {manufacturer}",
        f"NOC date: {noc_date}",
        f"Last update: {last_update}",
        f"Therapeutic class: {therapeutic_class}",
        f"Submission class: {submission_class}",
        f"Product type: {product_type}",
        f"Indication / reason for submission: {indication}",
        f"Country: {country}",
    ]

    # Include full JSON as backup context (helpful for rare fields),
    # but this will be chunked so we don't exceed token limits.
    full_json = json.dumps(entry, ensure_ascii=False)
    parts.append(f"Full NOC record (JSON): {full_json}")

    return "\n".join(parts)


def _preprocess_docs(data: List[Dict]) -> Tuple[List[str], List[str]]:
    """
    Prepare documents and IDs for Chroma embedding.

    - Each NOC entry is converted to a readable text blob.
    - The text is then split into CHUNK_SIZE character chunks to keep
      each embedding request well within token limits.
    - Returns:
        docs: list[str] of text chunks
        ids:  list[str] of unique IDs ('noc-<index>-chunk-<k>')
    """
    docs: List[str] = []
    ids: List[str] = []

    for i, entry in enumerate(data):
        base_text = _entry_to_base_text(entry)

        # chunk into manageable pieces
        for chunk_idx in range(0, len(base_text), CHUNK_SIZE):
            chunk = base_text[chunk_idx : chunk_idx + CHUNK_SIZE]
            doc_id = f"noc-{i}-chunk-{chunk_idx // CHUNK_SIZE}"
            docs.append(chunk)
            ids.append(doc_id)

    return docs, ids


# ---------------------------------------------------------------------------
# Custom embedding function using the NEW OpenAI client
# ---------------------------------------------------------------------------


class OpenAIEmbeddingAdapter:
    """
    Minimal embedding function for Chroma that uses the new OpenAI embeddings API.

    This avoids using the deprecated `openai.Embedding` interface and instead
    calls `client.embeddings.create(...)`.
    """

    def __init__(self, model_name: str):
        self.model_name = model_name

    def __call__(self, input: Documents) -> Embeddings:
        # `input` is a list of strings (documents)
        response = client.embeddings.create(
            model=self.model_name,
            input=input,
        )
        return [item.embedding for item in response.data]


# ---------------------------------------------------------------------------
# Chroma collection
# ---------------------------------------------------------------------------

_collection = None  # cached collection


def _get_collection():
    """
    Get or create the ChromaDB collection, and populate it if it's empty.

    - Uses a persistent client under chroma_store/
    - Embeds documents in small batches to avoid token-limit errors.
    - Limits the number of NOC entries so index build time is reasonable.
    """
    global _collection
    if _collection is not None:
        return _collection

    CHROMA_PATH.mkdir(parents=True, exist_ok=True)

    embedding_fn = OpenAIEmbeddingAdapter(EMBED_MODEL_NAME)

    chroma_client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    collection = chroma_client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn,
    )

    if collection.count() == 0:
        # Initial indexing – this may take a little while on first run
        print("[semantic] Building NOC semantic index (first run)...")
        data = _load_noc_data()

        # Limit dataset size so we don't try to embed everything
        if len(data) > MAX_ENTRIES:
            print(f"[semantic] Limiting NOC entries from {len(data)} to {MAX_ENTRIES} for faster indexing.")
            data = data[:MAX_ENTRIES]

        docs, ids = _preprocess_docs(data)

        for start in range(0, len(docs), BATCH_SIZE):
            end = start + BATCH_SIZE
            batch_docs = docs[start:end]
            batch_ids = ids[start:end]
            collection.add(documents=batch_docs, ids=batch_ids)

        print("[semantic] NOC semantic index built and stored in chroma_store/")

    _collection = collection
    return collection


# ---------------------------------------------------------------------------
# Public semantic search function
# ---------------------------------------------------------------------------


def semantic_drug_search(question: str) -> str:
    """
    Perform semantic search over the NOC dataset, then let the chat model
    answer the user's question using the retrieved context.
    """
    collection = _get_collection()

    result = collection.query(
        query_texts=[question],
        n_results=5,
    )

    documents = result.get("documents") or [[]]
    docs = documents[0] if documents else []

    if not docs:
        return (
            "I couldn't find relevant Notice of Compliance (NOC) entries for that question "
            "in the local dataset. You may want to try rephrasing your query or using "
            "the API-based service instead."
        )

    context = "\n\n---\n\n".join(docs)

    messages = [
        {
            "role": "system",
            "content": (
                "You are MedInsight, answering questions about Health Canada's "
                "Notice of Compliance (NOC) dataset. Use ONLY the provided context "
                "below; if something is not supported by the context, say that you "
                "cannot confirm it from the data.\n\n"
                "Summarize clearly in plain language for a Canadian healthcare "
                "professional, and mention that the source is the NOC dataset."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Question: {question}\n\n"
                f"Here are some relevant NOC entries:\n\n{context}"
            ),
        },
    ]

    completion = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=0.3,
    )

    return completion.choices[0].message.content.strip()
