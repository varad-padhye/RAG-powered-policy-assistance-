"""
policy_assistant/pipeline.py

Retrieval-augmented generation pipeline for the Policy Assistant.
Embedding and DB calls are stubbed with type-correct signatures;
replace stubs with real implementations before deploying.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import anthropic  # pip install anthropic

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Data types
# ──────────────────────────────────────────────

@dataclass
class PolicyChunk:
    chunk_id: str
    policy_id: str          # e.g. "POL-FIN-007"
    title: str
    section: str
    page: int
    text: str
    score: float = 0.0      # cosine similarity, populated after retrieval


@dataclass
class Citation:
    policy_id: str
    title: str
    section: str
    page: int


@dataclass
class AssistantResponse:
    answer_found: bool
    conflict_detected: bool
    answer: str
    citations: list[Citation]
    confidence: str          # "high" | "medium" | "low"
    escalation_note: Optional[str]
    latency_ms: int
    chunks_retrieved: int


# ──────────────────────────────────────────────
# Real implementations
# ──────────────────────────────────────────────

import os
import psycopg2
from pgvector.psycopg2 import register_vector
from sentence_transformers import SentenceTransformer, CrossEncoder

# Lazy-loaded global models
_embedding_model = None
_reranker_model = None


def get_embedding_model():
    global _embedding_model

    if _embedding_model is None:
        _embedding_model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2"
        )

    return _embedding_model


def get_reranker_model():
    global _reranker_model

    if _reranker_model is None:
        _reranker_model = CrossEncoder(
            "cross-encoder/ms-marco-MiniLM-L-6-v2"
        )

    return _reranker_model


def get_db_connection():
    """
    PostgreSQL connection using environment variables.

    Required env vars:
        PG_HOST
        PG_PORT
        PG_DB
        PG_USER
        PG_PASSWORD
    """

    conn = psycopg2.connect(
        host=os.environ["PG_HOST"],
        port=os.environ.get("PG_PORT", 5432),
        dbname=os.environ["PG_DB"],
        user=os.environ["PG_USER"],
        password=os.environ["PG_PASSWORD"],
    )

    register_vector(conn)

    return conn


def embed_query(text: str) -> list[float]:
    """
    Convert query text into embedding vector.
    """

    model = get_embedding_model()

    embedding = model.encode(text, normalize_embeddings=True)

    return embedding.tolist()


def vector_search(
    embedding: list[float],
    top_k: int = 8,
) -> list[PolicyChunk]:
    """
    pgvector similarity search.
    """

    sql = """
    SELECT
        chunk_id,
        policy_id,
        title,
        section,
        page,
        text,
        1 - (embedding <=> %s::vector) AS score
    FROM policy_chunks
    ORDER BY embedding <=> %s::vector
    LIMIT %s;
    """

    conn = get_db_connection()

    try:
        with conn.cursor() as cur:
            cur.execute(sql, (embedding, embedding, top_k))

            rows = cur.fetchall()

        chunks: list[PolicyChunk] = []

        for row in rows:
            chunks.append(
                PolicyChunk(
                    chunk_id=row[0],
                    policy_id=row[1],
                    title=row[2],
                    section=row[3],
                    page=row[4],
                    text=row[5],
                    score=float(row[6]),
                )
            )

        return chunks

    finally:
        conn.close()


def rerank(
    query: str,
    chunks: list[PolicyChunk],
    top_k: int = 5,
) -> list[PolicyChunk]:
    """
    Cross-encoder reranking.
    """

    if not chunks:
        return []

    reranker = get_reranker_model()

    pairs = [(query, chunk.text) for chunk in chunks]

    scores = reranker.predict(pairs)

    for chunk, score in zip(chunks, scores):
        chunk.score = float(score)

    ranked = sorted(
        chunks,
        key=lambda c: c.score,
        reverse=True,
    )

    return ranked[:top_k]


def log_eval(
    query: str,
    response: AssistantResponse,
    chunks: list[PolicyChunk],
) -> None:
    """
    Persist eval logs to PostgreSQL.
    """

    sql = """
    INSERT INTO eval_log (
        query,
        answer,
        citations_json,
        chunks_json,
        confidence,
        conflict,
        latency_ms,
        created_at
    )
    VALUES (
        %s,
        %s,
        %s,
        %s,
        %s,
        %s,
        %s,
        NOW()
    );
    """

    citations_payload = [
        {
            "policy_id": c.policy_id,
            "title": c.title,
            "section": c.section,
            "page": c.page,
        }
        for c in response.citations
    ]

    chunks_payload = [
        {
            "chunk_id": c.chunk_id,
            "policy_id": c.policy_id,
            "score": c.score,
            "text": c.text,
        }
        for c in chunks
    ]

    conn = get_db_connection()

    try:
        with conn.cursor() as cur:
            cur.execute(
                sql,
                (
                    query,
                    response.answer,
                    json.dumps(citations_payload),
                    json.dumps(chunks_payload),
                    response.confidence,
                    response.conflict_detected,
                    response.latency_ms,
                ),
            )

        conn.commit()

    finally:
        conn.close()
# ──────────────────────────────────────────────
# Context builder
# ──────────────────────────────────────────────

def build_context_block(chunks: list[PolicyChunk]) -> str:
    """
    Render retrieved chunks into the formatted context block
    that the system prompt's {context_block} slot expects.
    """
    parts: list[str] = []
    for i, chunk in enumerate(chunks, start=1):
        header = (
            f"[{chunk.policy_id} | {chunk.title} | "
            f"{chunk.section} | p.{chunk.page}]"
        )
        parts.append(f"Excerpt {i}:\n{header}\n{chunk.text}")
    return "\n\n".join(parts)


# ──────────────────────────────────────────────
# System prompt template
# ──────────────────────────────────────────────

SYSTEM_PROMPT_TEMPLATE = """\
You are an internal Policy Assistant.

You answer employee questions ONLY using the retrieved
policy excerpts below.

RETRIEVED CONTEXT
─────────────────
{context_block}

Rules:

1. Answer ONLY from retrieved policy excerpts.

2. Every factual statement must include inline citations:
   Example:
   "The meal limit is $150/day [POL-FIN-007]."

3. If information is missing:
   - respond:
     "This question is not covered in the retrieved policies."
   - set answer_found = false

4. If policies conflict:
   - show both answers with citations
   - set conflict_detected = true
   - recommend escalation to HR/Legal

5. Output format:

{
  "answer_found": true | false,
  "conflict_detected": true | false,
  "answer": "<natural language answer with inline [POL-XXX] citations>",
  "citations": [],
  "confidence": "high|medium|low"
}

6. Never use outside knowledge.

7. Return ONLY valid JSON.
"""



# ──────────────────────────────────────────────
# Generation
# ──────────────────────────────────────────────

def call_claude(
    query: str,
    context_block: str,
    client: anthropic.Anthropic,
    model: str = "claude-sonnet-4-20250514",
    max_tokens: int = 1024,
) -> dict:
    """
    Send the augmented prompt to Claude and parse the JSON response.

    Raises:
        ValueError  — if Claude returns malformed JSON
        RuntimeError — on API-level errors (caller handles retry)
    """
    system = SYSTEM_PROMPT_TEMPLATE.format(context_block=context_block)

    message = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": query}],
    )

    raw_text: str = message.content[0].text.strip()

    # Strip markdown fences if model wraps response despite instructions
    if raw_text.startswith("```"):
        raw_text = raw_text.split("```")[1]
        if raw_text.startswith("json"):
            raw_text = raw_text[4:]

    try:
        return json.loads(raw_text)
    except json.JSONDecodeError as exc:
        logger.error("Claude returned non-JSON: %s", raw_text[:300])
        raise ValueError(f"Malformed JSON from model: {exc}") from exc


# ──────────────────────────────────────────────
# Public entry point
# ──────────────────────────────────────────────

def answer_policy_question(
    query: str,
    client: anthropic.Anthropic,
    retrieval_top_k: int = 8,
    rerank_top_k: int = 5,
) -> AssistantResponse:
    """
    Full RAG pipeline: embed → retrieve → rerank → generate → log.

    Args:
        query:          The employee's natural-language question.
        client:         Authenticated Anthropic client.
        retrieval_top_k: How many chunks to pull from pgvector.
        rerank_top_k:   How many chunks to keep after cross-encoder rerank.

    Returns:
        AssistantResponse (always returned; never raises to caller)

    Error contract:
        - Embedding failure    → return refusal response, log WARN
        - Retrieval failure    → return refusal response, log ERROR
        - Claude API failure   → retry once with 2s back-off, then refusal
        - JSON parse failure   → return refusal response, log ERROR
    """
    t0 = time.monotonic()

    # ── 1. Embed query ────────────────────────────────────────────────
    try:
        embedding = embed_query(query)
    except Exception as exc:
        logger.warning("Embedding failed: %s", exc)
        return _refusal_response(t0, chunks_retrieved=0)

    # ── 2. Vector search ──────────────────────────────────────────────
    try:
        raw_chunks = vector_search(embedding, top_k=retrieval_top_k)
    except Exception as exc:
        logger.error("Vector search failed: %s", exc)
        return _refusal_response(t0, chunks_retrieved=0)

    if not raw_chunks:
        logger.info("No chunks retrieved for query: %r", query[:80])
        return _refusal_response(t0, chunks_retrieved=0)

    # ── 3. Re-rank ────────────────────────────────────────────────────
    try:
        ranked_chunks = rerank(query, raw_chunks, top_k=rerank_top_k)
    except Exception as exc:
        logger.warning("Re-rank failed, falling back to bi-encoder order: %s", exc)
        ranked_chunks = raw_chunks[:rerank_top_k]

    # ── 4. Build context and call Claude ─────────────────────────────
    context_block = build_context_block(ranked_chunks)

    raw_response: dict = {}
    for attempt in (1, 2):
        try:
            raw_response = call_claude(query, context_block, client)
            break
        except ValueError as exc:
            # JSON parse error — no point retrying
            logger.error("JSON parse failure on attempt %d: %s", attempt, exc)
            return _refusal_response(t0, len(ranked_chunks))
        except Exception as exc:
            logger.warning("Claude API error attempt %d: %s", attempt, exc)
            if attempt == 2:
                return _refusal_response(t0, len(ranked_chunks))
            time.sleep(2)

    # ── 5. Parse structured response ─────────────────────────────────
    try:
        response = AssistantResponse(
            answer_found=bool(raw_response.get("answer_found", False)),
            conflict_detected=bool(raw_response.get("conflict_detected", False)),
            answer=str(raw_response.get("answer", "")),
            citations=[
                Citation(**c) for c in raw_response.get("citations", [])
            ],
            confidence=str(raw_response.get("confidence", "low")),
            escalation_note=raw_response.get("escalation_note"),
            latency_ms=int((time.monotonic() - t0) * 1000),
            chunks_retrieved=len(ranked_chunks),
        )
    except (KeyError, TypeError) as exc:
        logger.error("Response schema mismatch: %s — raw: %s", exc, raw_response)
        return _refusal_response(t0, len(ranked_chunks))

    # ── 6. Async eval logging ─────────────────────────────────────────
    try:
        log_eval(query, response, ranked_chunks)
    except Exception as exc:
        # Never let eval logging break a successful response
        logger.warning("eval_log failed silently: %s", exc)

    return response


def _refusal_response(t0: float, chunks_retrieved: int) -> AssistantResponse:
    """Canonical fallback when any pipeline stage fails."""
    return AssistantResponse(
        answer_found=False,
        conflict_detected=False,
        answer=(
            "I wasn't able to retrieve a reliable answer from the policy "
            "documents right now. Please contact HR, IT, or Finance directly, "
            "or check the company intranet."
        ),
        citations=[],
        confidence="low",
        escalation_note="System could not complete retrieval or generation.",
        latency_ms=int((time.monotonic() - t0) * 1000),
        chunks_retrieved=chunks_retrieved,
    )
    
# ──────────────────────────────────────────────
# Run directly
# ──────────────────────────────────────────────

if __name__ == "__main__":

    import os
    import anthropic

    # Create Anthropic client
    client = anthropic.Anthropic(
        api_key=os.environ["ANTHROPIC_API_KEY"]
    )

    print("Policy Assistant Ready")
    print("Type 'exit' to quit\n")

    while True:

        query = input("Ask policy question: ").strip()

        if query.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        try:
            response = answer_policy_question(
                query=query,
                client=client,
            )

            print("\n=== RESPONSE ===")
            print(f"Answer Found      : {response.answer_found}")
            print(f"Confidence        : {response.confidence}")
            print(f"Conflict Detected : {response.conflict_detected}")
            print(f"Latency           : {response.latency_ms} ms")
            print(f"Chunks Retrieved  : {response.chunks_retrieved}")

            print("\nAnswer:")
            print(response.answer)

            if response.citations:
                print("\nCitations:")
                for c in response.citations:
                    print(
                        f"- {c.policy_id} | "
                        f"{c.title} | "
                        f"{c.section} | "
                        f"p.{c.page}"
                    )

            if response.escalation_note:
                print("\nEscalation Note:")
                print(response.escalation_note)

            print("\n" + "=" * 60 + "\n")

        except KeyboardInterrupt:
            print("\nInterrupted. Exiting...")
            break

        except Exception as exc:
            print(f"\nUnexpected error: {exc}\n")
