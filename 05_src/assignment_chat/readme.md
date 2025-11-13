# MedInsight – Compliance-Aware Conversational Agent  
### Deploying AI – Assignment 2  
**Author:** Syyeda Kashfa Azim  
**Repository Branch:** `assignment-2`  

---

## 🔎 1. Project Overview  
MedInsight is a conversational AI system designed to support regulatory and medication-related queries in a Canadian context. The system integrates three major services:  
- an FDA drug label lookup service,  
- a semantic search over the Canadian Notice of Compliance (NOC) dataset, and  
- a high-level compliance reasoning service.  
A unified chat interface built with Gradio orchestrates these services, routes queries intelligently, and delivers a seamless user experience.

The key design goals:  
- Use at least **three services** as required by the assignment.  
- Use an API (for the FDA lookup service).  
- Use a semantic search / vector search back-end (for NOC).  
- Provide a distinct personality for the chat interface, maintain chat memory, and implement guardrails to restrict certain topics.

---

## 🛠 2. System Architecture  
### 2.1 High-Level Workflow  
1. User enters a message into the Gradio chat UI.  
2. A router module examines the message to determine whether it belongs to:  
   - the `/api <drug>` command (FDA service),  
   - the `/semantic <question>` command (NOC semantic service),  
   - the `/compliance <scenario>` command (compliance reasoning service), or  
   - a normal conversational fallback.  
3. The selected service executes its back-end logic:  
   - API service calls the openFDA API, transforms results.  
   - Semantic service ensures the NOC dataset is loaded, builds embeddings if needed, queries ChromaDB, then synthesizes an answer via OpenAI chat.  
   - Compliance service invokes the function-calling tool to assess risk and returns a summary.  
4. The reply is returned to the Gradio front-end as a “messages”-type chat item.

### 2.2 Directory Structure  
# MedInsight – Compliance-Aware Conversational Agent  
### Deploying AI – Assignment 2  
**Author:** Syyeda Kashfa Azim  
**Repository Branch:** `assignment-2`  

---

## 🔎 1. Project Overview  
MedInsight is a conversational AI system designed to support regulatory and medication-related queries in a Canadian context. The system integrates three major services:  
- an FDA drug label lookup service,  
- a semantic search over the Canadian Notice of Compliance (NOC) dataset, and  
- a high-level compliance reasoning service.  
A unified chat interface built with Gradio orchestrates these services, routes queries intelligently, and delivers a seamless user experience.

The key design goals:  
- Use at least **three services** as required by the assignment.  
- Use an API (for the FDA lookup service).  
- Use a semantic search / vector search back-end (for NOC).  
- Provide a distinct personality for the chat interface, maintain chat memory, and implement guardrails to restrict certain topics.

---

## 🛠 2. System Architecture  
### 2.1 High-Level Workflow  
1. User enters a message into the Gradio chat UI.  
2. A router module examines the message to determine whether it belongs to:  
   - the `/api <drug>` command (FDA service),  
   - the `/semantic <question>` command (NOC semantic service),  
   - the `/compliance <scenario>` command (compliance reasoning service), or  
   - a normal conversational fallback.  
3. The selected service executes its back-end logic:  
   - API service calls the openFDA API, transforms results.  
   - Semantic service ensures the NOC dataset is loaded, builds embeddings if needed, queries ChromaDB, then synthesizes an answer via OpenAI chat.  
   - Compliance service invokes the function-calling tool to assess risk and returns a summary.  
4. The reply is returned to the Gradio front-end as a “messages”-type chat item.
 
### 2.2 Directory Structure (Mermaid Diagram)

```mermaid
graph TD
    A[05_src] --> B[assignment_chat]
    B --> C[app.py]
    B --> D[router.py]
    B --> E[__init__.py]
    B --> F[services]
    B --> G[chroma_store/]

    F --> H[api_call.py]
    F --> I[semantic.py]
    F --> J[compliance_check.py]
    F --> K[guardrails.py]
    F --> L[data/]

    L --> M[noc_drug_products.json]


_Internal services may call APIs, build embeddings, use function-calling, etc._

---
## ** 3. 🎯 Services Implemented **

### ** 3.1 FDA Drug Label Lookup (`/api <drug>`) **
- Uses the openFDA drug label API.  
- Transforms the raw API JSON into a natural-language summary.  
- Example user query: `/api aspirin`  
- Outputs include: generic name, manufacturer, indications, warnings, adverse reactions.

### 3.2 NOC Semantic Search (`/semantic <question>`)  
- Automatically checks for the local NOC dataset; if missing or invalid, it downloads the JSON file from the Health Canada open-government portal.  
- Pre-processes all entries: converts each JSON record into text, chunks large entries (via `_preprocess_docs()`), and builds embeddings using OpenAI.  
- Uses ChromaDB to store vectors persistently.  
- Supports natural-language search queries.  
- Example: `/semantic Which NOC entries mention breast cancer?`  
- Returns results with NOC numbers, dates, manufacturers, therapeutic class, and a short summary.

### 3.3 Compliance Reasoning Service (`/compliance <scenario>`)  
- Uses OpenAI function-calling to invoke `assess_regulatory_risk()` tool with inferred parameters (product type, intended use, target population, jurisdiction).  
- Returns a structured summary: risk level, contributing factors, regulatory checklist.  
- Example: `/compliance injectable oncology drug in children in Canada`  
- Outcome: high-level compliance advice (non-medical).

---

## ✅ 4. Implementation Highlights  
- Environment: Python 3.12  
- Chat interface: Gradio (with `type="messages"` to avoid tuple format warning)  
- Vector store: ChromaDB persistent client  
- Embeddings: OpenAI embedding API (new version, no deprecated calls)  
- Guardrails: restrict discussions about cats, dogs, horoscopes, Taylor Swift, and system prompt modifications  
- Memory: chat history maintained in Gradio session — part of short-term conversational memory  
- Tools: function-calling used in compliance service, API calls used in lookup service

---

## 📂 5. Dataset  
- Primary dataset: `noc_drug_products.json` — the Notice of Compliance (NOC) drug products database from Health Canada (English JSON format, ~20 MB).  
- Storage: under `services/data/`.  
- Pre-processing: chunking of records to keep embedding input size manageable.  
- Indexing: limited to `MAX_ENTRIES = 1500` to ensure build time is reasonable; stored in `services/data/`.

### Embeddings & ChromaDB Storage
For the semantic search service, I use the Health Canada Notice of Compliance (NOC) dataset stored in:
- `05_src/assignment_chat/services/data/noc_drug_products.json`
This JSON file is **under the 40 MB limit** requested in the assignment.
From this source file, the app automatically:
1. Loads and preprocesses the NOC records.
2. Builds OpenAI embeddings.
3. Stores them in a **persistent ChromaDB index** under `05_src/assignment_chat/chroma_store/`.
However, to keep the repository size reasonable and to match the course expectations, I **do not commit the ChromaDB index directory**. Instead:
- `chroma_store/` is listed in `.gitignore`.
- Only the source dataset (`noc_drug_products.json`) and the embedding/semantic-search code (`semantic.py`) are included.
- When the reviewer runs the app, the semantic index will be rebuilt locally on first `/semantic` query (as indicated by log messages such as `[semantic] Building NOC semantic index (first run)...`).

---

## 🧩 6. Limitations & Assumptions  
- The system does **not** provide medical diagnosis or treatment recommendations; it only provides informational or regulatory-style responses.  
- Compliance reasoning is generic; not specialized legal advice.  
- The NOC dataset snapshot is static and may not include the most recent updates.  
- Embeddings and semantic search are approximate; results should be interpreted carefully.  
- The system uses a subset of entries for performance reasons (first run build time).  
- User queries outside the regulatory domain (e.g., pets, astrology, entertainment) are blocked by guardrails.

---

## 🔭 7. Future Work  
- Expand dataset to include the full NOC dataset and possibly French entries.  
- Integrate additional regulatory data sources (EMA, FDA, Health Canada other datasets).  
- Enable incremental updates so the index remains fresh without full rebuilds.  
- Add UI capabilities: dashboard view of query analytics, approval trends, visualization of vault compliance checklists.  
- Enhance memory management: summarise older chat turns, discard extraneous context to maintain efficient token usage.

---

## 📌 8. How to Run Locally  
1. Clone the repository and switch to branch `assignment-2`.  
2. Create and activate virtual environment:
    ```bash
    source deploying-ai-env/bin/activate
    ```
3. Add your OpenAI API key in `05_src/.secrets`:
    ```
    OPENAI_API_KEY=sk-your_key_here
    ```
4. Start the app:
    ```bash
    python -m 05_src.assignment_chat.app
    ```
5. Open browser at `http://127.0.0.1:7860` and start chatting!

---

## ℹ️ 9. Example Usage  
- `/api ibuprofen` → returns an FDA label summary  
- `/semantic NOC approvals oncology 2019 Canada` → returns NOC entries  
- `/compliance off-label pediatric drug in Canada` → compliance reasoning  

---

## 🎓 10. Acknowledgements  
- University of Toronto – Data Sciences Institute  
- Deploying AI course instructors & session facilitators  
- OpenAI for LLM embeddings and function-calling  
- Health Canada for open regulatory datasets  

---

## 📜 11. License  
This project is developed for educational use under the **Deploying AI (University of Toronto DSI)** course.  
It is not intended for clinical, commercial, or production deployment.

---

