# RAG-Powered Policy Assistant — Technical Spec

# 1. Architecture Sketch

## Pipeline Overview

```text
POLICY PDFs
    ↓
PDF Parsing (pdfplumber)
    ↓
Chunking (512 tokens + overlap)
    ↓
Embeddings (sentence-transformers/all-MiniLM-L6-v2)
    ↓
PostgreSQL + pgvector
    ↓
User Query
    ↓
Query Embedding
    ↓
Vector Search (Top-K Retrieval)
    ↓
Cross-Encoder Re-ranking
    ↓
Claude Sonnet 4
    ↓
Structured Answer + Citations
    ↓
Evaluation Layer
```

---

## Tech Choices

| Component    | Technology                             | Justification                                                  |
| ------------ | -------------------------------------- | -------------------------------------------------------------- |
| PDF Parsing  | pdfplumber                             | Handles tables and multi-column PDFs better than PyPDF2        |
| Embeddings   | sentence-transformers/all-MiniLM-L6-v2 | Fast CPU inference with strong semantic similarity performance |
| Vector Store | PostgreSQL + pgvector                  | Easy operational management and reliable metadata consistency  |
| Re-ranking   | cross-encoder/ms-marco-MiniLM-L-6-v2   | Improves retrieval precision before generation                 |
| LLM          | Claude Sonnet 4                        | Strong instruction-following and grounded generation           |
| Backend      | Python 3.10+                           | Mature AI ecosystem and production support                     |

---

# 2. Answering Prompt

```text
SYSTEM PROMPT

You are an internal Policy Assistant.

Rules:

1. Answer ONLY from retrieved policy excerpts.
2. Every factual statement must include inline citations:
   Example: "The meal limit is $150/day [POL-FIN-007]."

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
  "answer_found": true,
  "conflict_detected": false,
  "answer": "...",
  "citations": [],
  "confidence": "high|medium|low"
}

6. Never use outside knowledge.
```

---

# 3. Implementation Skeleton

```
Implemented skeleton is showm in different file name pipeline.py
```
---

# 4. Evals

| Category                | Test Input                                                | Success Criterion                             | Grading      |
| ----------------------- | --------------------------------------------------------- | --------------------------------------------- | ------------ |
| Exact Fact Recall       | “What is the international meal limit?”                   | Returns “$150” + correct citation             | Auto         |
| Multi-Doc Synthesis     | “Can new hires work remotely on personal laptops?”        | Mentions both 90-day rule and MDM requirement | LLM-as-judge |
| Refusal                 | “What is sabbatical policy?”                              | Refuses correctly with no hallucination       | Auto         |
| Citation Correctness    | “Do I need receipts?”                                     | Correct citation to POL-FIN-007 only          | Auto         |
| Hallucination Detection | “Domestic travel meal limit?”                             | Does not invent dollar amount                 | LLM-as-judge |
| Conflict Handling       | “Can I use my personal laptop?” with conflicting policies | Shows both policies + escalation              | Mixed        |

---

# 5. Failure Recovery

| Failure Mode                                    | Monitoring Signal              | Fallback                              |
| ----------------------------------------------- | ------------------------------ | ------------------------------------- |
| Stale policy chunks                             | Policy version hash mismatch   | Re-ingest updated documents           |
| Semantic mismatch (“BYOD” vs “personal device”) | Low retrieval similarity score | Query expansion with synonym mapping  |
| Undetected policy conflicts                     | Conflict registry mismatch     | Force retrieval from related policies |

---

# 6. Client Message

Hi,

We’ve built the first version of the Policy Assistant using a Retrieval-Augmented Generation (RAG) architecture. Employees can ask natural-language policy questions and receive grounded answers with inline citations to the original policy documents.

The system refuses to answer when the information is not present in retrieved policies, reducing hallucination risk. It also detects explicit policy conflicts and recommends escalation instead of guessing.

Version 1 is optimized for correctness and operational simplicity, but it may struggle with acronym mismatches (“BYOD” vs “personal device”), scanned PDFs, and hidden cross-policy conflicts.

To improve Version 2, strong document hygiene is important. Policies should have stable IDs, clear section headings, version tracking, and an assigned owner. Future improvements can include hybrid retrieval, real-time document syncing, role-based access control, and feedback-driven continuous learning.
