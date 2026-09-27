
# Zepto Support Assistant

## Module 3 - GenAI Support Assistant

This module implements a small GenAI support assistant for Zepto policy
questions: an 8-document policy corpus, embedded locally and retrieved via
ChromaDB, orchestrated through a LangGraph intent router, wrapped in a
FastAPI service, and containerized with Docker.

Stack: Sentence Transformers (`all-MiniLM-L6-v2`), ChromaDB, LangGraph,
FastAPI + Pydantic, and a deterministic `MOCK_LLM` mode as the graded baseline.

## Setup

```bash
pip install -r requirements.txt
python ingest.py        # builds the ChromaDB index from docs/
uvicorn main:app --host 0.0.0.0 --port 7860
```

Or with Docker:

```bash
docker build -t zepto-support-assistant .
docker run -p 7860:7860 zepto-support-assistant
```

API reachable at `http://localhost:7860/ask`.

## MOCK_LLM toggle

Every generation step is gated behind the `MOCK_LLM` environment variable
(default `"1"` = mock mode, the graded baseline):

- **Default (`MOCK_LLM=1`):** no LLM call anywhere. `classify_intent` uses a
  keyword heuristic; `retrieve_and_answer` returns
  `f"Based on the retrieved context: {top_chunk[:200]}"` from the top
  ChromaDB match; `direct_answer` returns a fixed canned string. The
  `AskResponse` is populated deterministically, `confidence` fixed at `1.0`.
- **`MOCK_LLM=0` (optional, ungraded extension):** the same nodes would call
  a real LLM instead, using the structured prompt template
  (role/context/task/format/length + negative constraint + few-shot example)
  defined for this module.

Retrieval (embedding the query + querying ChromaDB) always runs for real in
both modes, since it needs no API key or network access.

## RAG pipeline (ingestion -> embedding -> retrieval -> generation)

1. **Ingestion** - `docs/doc_01.txt` ... `doc_08.txt` hold Zepto's policy text.
2. **Chunking** - one chunk per document (`{doc_id}_chunk_0` / `{doc_id}`),
   sufficient given the short document length.
3. **Embedding** - `ingest.py` embeds each chunk locally with
   `all-MiniLM-L6-v2` (no API key, no network call after first model download).
4. **Storage** - embeddings + text + source metadata are stored in a
   persistent ChromaDB collection (`zepto_policies`) under `chroma_store/`.
5. **Retrieval** - the `retrieve_and_answer` LangGraph node (in `graph.py`)
   embeds the incoming query and retrieves the top-3 most similar chunks.
6. **Generation** - in mock mode, the same node builds the answer directly
   from the top retrieved chunk; `direct_answer` skips retrieval entirely.
7. **Routing** - `classify_intent` classifies the query via keyword
   heuristic; a conditional LangGraph edge routes to `retrieve_and_answer`
   or `direct_answer`.
8. **Response** - `main.py`'s `/ask` endpoint validates the final state
   against the `AskResponse` Pydantic model and returns it as JSON.

## LangGraph nodes

- **classify_intent** - checks the lowercased query against policy keywords
  (delivery, return, refund, membership, tracking, cancel, gift card,
  support hours) to classify as `policy_question` or `general_question`.
- **retrieve_and_answer** - embeds the query, retrieves top-3 chunks from
  ChromaDB, builds the mock answer from the top chunk, returns chunk ids as
  `sources`.
- **direct_answer** - returns a fixed string, no retrieval, no LLM call.

## Structured response

```json
{
  "answer": "string",
  "sources": ["chunk_id"],
  "confidence": 1.0
}
```

## Example API calls (MOCK_LLM left at default)

**Example 1 - policy question (routes to retrieve_and_answer):**

Request:
```json
{"query": "What is your delivery fee?"}
```

Response:
```json
{
  "answer": "Based on the retrieved context: Delivery Policy: Zepto delivers grocery and household essentials to serviceable pin codes within 10 to 30 minutes of order confirmation, depending on the customer's delivery zone and current order vol",
  "sources": ["doc_01_chunk_0", "doc_05_chunk_0", "doc_02_chunk_0"],
  "confidence": 1.0
}
```

**Example 2 - general question (routes to direct_answer):**

Request:
```json
{"query": "Tell me a joke"}
```

Response:
```json
{
  "answer": "I can only answer questions about Zepto policies right now.",
  "sources": [],
  "confidence": 1.0
}
```

## Design decisions

- **One chunk per document:** each policy document is a short single
  paragraph, so per-document chunking avoids unnecessary fragmentation.
- **Keyword-based intent routing in mock mode:** keeps the graded baseline
  fully deterministic and network-free while still exercising a real
  conditional-edge graph.
- **Persistent ChromaDB store (`chroma_store/`):** embeddings are computed
  once via `ingest.py` and reused by `main.py` at startup.
- **Fixed `confidence=1.0` in mock mode:** there is no LLM output to be
  uncertain about, so confidence is a deterministic constant.
