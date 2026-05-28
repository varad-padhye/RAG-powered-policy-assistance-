# RAG-Powered Policy Assistant — v1 Technical Spec

---

## 1. Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        INGESTION PIPELINE                           │
│                        (runs offline / on upload)                   │
│                                                                     │
│  50x PDFs  ──▶  pdfplumber  ──▶  Chunker  ──▶  sentence-           │
│  (HR / IT /      (text +           (512 tok,      transformers      │
│   Finance)        tables)          128 overlap)   all-MiniLM-L6-v2  │
│                                        │               │            │
│                                        ▼               ▼            │
│                                   Metadata        768-dim           │
│                                   (doc_id,        embedding         │
│                                    page, sec)         │            │
│                                        └───────────────┘            │
│                                                │                    │
│                                                ▼                    │
│                                    ┌──────────────────┐            │
│                                    │  PostgreSQL +     │            │
│                                    │  pgvector         │            │
│                                    │  chunks table     │            │
│                                    └──────────────────┘            │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                        QUERY PIPELINE (runtime)                     │
│                                                                     │
│  User question                                                      │
│       │                                                             │
│       ▼                                                             │
│  same embedding model  ──▶  pgvector ANN search                    │
│  (all-MiniLM-L6-v2)          (cosine, top-k=8)                     │
│                                    │                                │
│                                    ▼                                │
│                            Re-rank with cross-encoder               │
│                            (cross-encoder/ms-marco-MiniLM-L-6-v2)  │
│                                    │                                │
│                                    ▼                                │
│                            Top-5 chunks + metadata                  │
│                                    │                                │
│                                    ▼                                │
│                       ┌────────────────────────┐                   │
│                       │  Claude claude-sonnet-4  │                   │
│                       │  (system prompt below)   │                   │
│                       └────────────────────────┘                   │
│                                    │                                │
│                                    ▼                                │
│                       Structured JSON response                      │
│                       (answer + citations + flags)                  │
│                                    │                                │
│                                    ▼                                │
│                       ┌────────────────────────┐                   │
│                       │  Eval layer (async)     │                   │
│                       │  - faithfulness score   │                   │
│                       │  - citation check       │                   │
│                       │  - logged to pg table   │                   │
│                       └────────────────────────┘                   │
└─────────────────────────────────────────────────────────────────────┘
```

### Tech Choice Justifications

| Component | Choice | Reason |
|---|---|---|
| **PDF extraction** | `pdfplumber` | Handles tables and multi-column layouts better than PyPDF2; pure Python, no Java dependency |
| **Embeddings** | `sentence-transformers/all-MiniLM-L6-v2` | 768-dim, runs on CPU, sub-100ms per chunk; strong on semantic similarity for policy-style prose |
| **Vector store** | PostgreSQL + `pgvector` | Client already has Postgres expertise; single operational dependency; ACID guarantees on metadata updates when policies change |
| **Re-ranker** | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Cross-encoders outperform bi-encoders at re-ranking; adds ~30ms, materially reduces false-positive retrievals |
| **Generation** | Claude claude-sonnet-4 | Long context, strong instruction-following for citation format; low hallucination rate on grounded tasks |
| **Eval storage** | Same Postgres instance, `eval_log` table | Keeps infra simple; enables trend queries across policy versions |

---

## 2. Generation System Prompt

```
SYSTEM PROMPT — Policy Assistant v1
=====================================

You are an internal Policy Assistant for [Company]. Your sole job is to
answer employee questions using the retrieved policy excerpts provided
below. You do not use outside knowledge.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RETRIEVED CONTEXT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{context_block}
  — Each excerpt is prefixed with its Policy ID, document title,
    section heading, and page number.
  — Example prefix:
    [POL-FIN-007 | Travel & Expense Policy | §4.2 International Limits | p.11]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RULES — follow every rule exactly
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. GROUNDING
   Answer only from the retrieved excerpts. If the excerpts do not
   contain enough information to answer, set "answer_found": false and
   use the refusal template below. Never infer, extrapolate, or
   supplement with general knowledge.

2. INLINE CITATIONS
   Every factual claim must be followed immediately by its Policy ID
   in square brackets, e.g. "The daily limit is $75 [POL-FIN-007]."
   If a single claim is supported by two excerpts, cite both:
   "Approval is required [POL-FIN-007, POL-HR-003]."
   Do not group all citations at the end.

3. CONFLICT HANDLING
   If two excerpts give contradictory answers:
   - State both versions with their respective citations.
   - Set "conflict_detected": true.
   - Recommend the employee contact the policy owner (listed in the
     document metadata if available) or HR/Legal for a ruling.
   - Do NOT pick a side or guess which policy supersedes the other.

4. REFUSAL
   If the question cannot be answered from the excerpts, respond
   exactly:
     "This question isn't covered by the policy documents I have
      access to. Please contact [HR/IT/Finance] or consult the
      company intranet for guidance."
   Set "answer_found": false. Do not add speculation.

5. OUTPUT FORMAT
   Always return valid JSON matching this schema exactly:

   {
     "answer_found": true | false,
     "conflict_detected": true | false,
     "answer": "<natural language answer with inline [POL-XXX] citations>",
     "citations": [
       {
         "policy_id": "POL-FIN-007",
         "title": "Travel & Expense Policy",
         "section": "§4.2 International Limits",
         "page": 11
       }
     ],
     "confidence": "high" | "medium" | "low",
     "escalation_note": "<string or null>"
   }

   "confidence" rules:
     high   — answer is explicit and unambiguous in the excerpts
     medium — answer requires minor inference across excerpts
     low    — excerpts are tangentially relevant; recommend verification

   "escalation_note" — populate only when conflict_detected is true or
   confidence is low; otherwise null.

6. TONE
   Professional, neutral, concise. Employees are not lawyers; avoid
   jargon. If a policy number alone is confusing, add the document's
   plain-language name in parentheses.
```

---

## 3. Implementation Skeleton

```
Skeleton code is show seperatly on github
```

## 4. Evals

### Mock policy snippets for eval context

```
[POL-FIN-007 | Travel & Expense Policy | §4.2 International Limits | p.11]
Employees on international assignments may claim up to $150 USD per day for
meals and incidentals. Receipts are required for any single expense > $25.

[POL-IT-002 | Acceptable Use Policy | §2.1 Personal Devices | p.4]
Personal laptops may be used for work provided they are enrolled in the
company MDM solution and meet minimum OS version requirements (Appendix A).

[POL-HR-011 | Remote Work Policy | §3.0 Eligibility | p.7]
Employees must have completed 90 days of employment before applying for
a permanent remote work arrangement. Manager approval is required.
```

---

### Eval 1 — Exact Fact Recall

| Field | Value |
|---|---|
| **Category** | Exact fact recall |
| **Input** | *"What is the daily meal and incidentals limit for international travel?"* Context: POL-FIN-007 only |
| **Success criterion** | `answer` contains "$150" or "150 USD" AND `citations` includes `POL-FIN-007`. No extra dollar figure fabricated. |
| **Grading** | ✅ **Auto-gradeable** — regex match on "$150" + citation field check |

---

### Eval 2 — Multi-Document Synthesis

| Field | Value |
|---|---|
| **Category** | Multi-document synthesis |
| **Input** | *"I'm a new hire, 3 weeks in. Can I work remotely from my personal laptop?"* Context: POL-HR-011 + POL-IT-002 |
| **Success criterion** | Response correctly identifies **two** separate blockers: (a) 90-day tenure requirement [POL-HR-011], (b) MDM enrollment required for personal laptops [POL-IT-002]. Both citations present. Answer does not say "yes" to either. |
| **Grading** | ⚠️ **LLM-as-judge** — ask a grader model to verify both blockers are named with correct citations |

---

### Eval 3 — Refusal of Out-of-Scope Questions

| Field | Value |
|---|---|
| **Category** | Refusal |
| **Input** | *"What is the company's policy on sabbaticals?"* Context: none of the 3 mock policies mention sabbaticals |
| **Success criterion** | `answer_found` is `false`, `answer` contains no dollar figures or policy numbers, response directs user to HR or intranet, no hallucinated policy cited |
| **Grading** | ✅ **Auto-gradeable** — assert `answer_found == false` + assert no `[POL-*]` IDs in `answer` |

---

### Eval 4 — Citation Correctness

| Field | Value |
|---|---|
| **Category** | Citation correctness |
| **Input** | *"Do I need receipts for international expenses?"* Context: POL-FIN-007 (receipts required > $25) |
| **Success criterion** | Citation in response points to `POL-FIN-007`, section `§4.2`, page `11`. **No** citation to POL-IT-002 or POL-HR-011. The claim "receipts required for expenses over $25" is present. |
| **Grading** | ✅ **Auto-gradeable** — exact match on `citations[].policy_id`, `citations[].page`, and substring match on "$25" in answer |

---

### Eval 5 — Hallucination Detection

| Field | Value |
|---|---|
| **Category** | Hallucination detection |
| **Input** | *"What's the daily domestic travel meal limit?"* Context: POL-FIN-007 mentions **international** limit only; domestic limit is not stated |
| **Success criterion** | Response does NOT state a domestic dollar figure. Either `answer_found = false` or the answer explicitly says "the policy only specifies the international limit" with `confidence` ≤ "medium". No invented figure like "$75" or "$100". |
| **Grading** | ⚠️ **LLM-as-judge** — grader checks: "Does the answer state any specific domestic dollar amount not present in the provided context?" |

---

### Eval 6 — Conflict Handling

| Field | Value |
|---|---|
| **Category** | Conflict handling |
| **Input** | *"Can I use my personal laptop for work?"* Context: POL-IT-002 says yes (with MDM) **+** a synthetic conflicting snippet: `[POL-SEC-001 | Security Policy | §5.1 | p.2] "Personal devices are not permitted to access company systems or data under any circumstances."` |
| **Success criterion** | `conflict_detected = true`, both POL-IT-002 and POL-SEC-001 are cited, response does NOT recommend one over the other, `escalation_note` is populated and names HR/Legal/policy owner as the resolution path |
| **Grading** | ✅ **Auto-gradeable** for structure (`conflict_detected`, both IDs in citations). ⚠️ **Human review** for tone — verify no implicit side-taking |

---

## 5. Failure Recovery

### Failure Mode 1 — Stale Policy Chunks After Document Update

**What it is:** A policy PDF is updated (e.g., the international meal limit goes from $150 to $175) but the old chunks remain in pgvector. The system confidently cites the old figure with high confidence.

**Why it's insidious:** No error is thrown. Retrieval "succeeds." The wrong answer is delivered with a real policy citation, so employees trust it.

**Monitoring signal:**
- Each policy document gets a `version_hash` (SHA-256 of the PDF bytes) stored alongside chunks.
- A nightly job diffs `version_hash` against the live documents on SharePoint/GDrive.
- Alert fires if `updated_at` on the source doc is newer than `ingested_at` on any of its chunks.

**Fallback:**
- Stale chunks get a `stale = true` flag immediately; they are excluded from retrieval until re-ingestion completes.
- If re-ingestion is still in progress, the query pipeline surfaces a banner: *"This policy is being updated. Please verify with the policy owner before acting."*

---

### Failure Mode 2 — Query-Chunk Semantic Mismatch (Vocabulary Gap)

**What it is:** Employees use colloquial language ("bring my own laptop," "BYOD") while the policy uses formal language ("personal devices enrolled in MDM"). The embedding similarity is low; the right chunk lands outside top-k and a worse chunk wins, producing a plausible but wrong answer.

**Why it's insidious:** The system returns `answer_found = true` with a real citation — just the wrong one. No retrieval error is logged.

**Monitoring signal:**
- Log the cosine similarity score of the top-ranked chunk after re-ranking.
- Alert when the average top-1 score across a rolling 100-query window drops below 0.70.
- Flag individual queries where top-1 score < 0.60 for human review in the eval dashboard.

**Fallback:**
- For low-score queries (< 0.60), force `confidence = "low"` regardless of Claude's output and populate `escalation_note`.
- Maintain a synonym/alias table (e.g., "BYOD → personal device, personal laptop") that expands queries before embedding. Seed from the first 30 days of user queries.

---

### Failure Mode 3 — Cross-Policy Conflict Goes Undetected

**What it is:** Two policies genuinely conflict but the conflicting chunks don't both appear in the top-k results (they may live in rarely-retrieved documents). The assistant picks the higher-ranked chunk and answers confidently — `conflict_detected = false` — while an authoritative contradicting policy goes unseen.

**Why it's insidious:** The system doesn't know what it didn't retrieve. No signal is emitted. The employee acts on a one-sided answer.

**Monitoring signal:**
- Maintain a hand-curated **conflict registry** table: `(topic_tag, [policy_id_A, policy_id_B])` for known tension areas (e.g., BYOD appears in both IT Acceptable Use and Information Security).
- At query time, after retrieval, check if the retrieved set covers only one side of a registered conflict pair for the detected topic tags.
- If yes: force-fetch the missing policy's anchor chunk and inject it, then re-run generation.

**Fallback:**
- For topics in the conflict registry, always retrieve at least one chunk from each registered conflicting document, bypassing pure ANN ranking.
- Add a post-generation check: if the response cites a policy known to be in a conflict pair but doesn't cite the counterpart, either inject a disclaimer or trigger re-generation with both chunks included.

---

## 6. Client Message



Hi,

We've built and scoped the foundation of your Policy Assistant. Employees can now ask natural-language questions ("What's the international meal limit?") and get answers with exact citations back to the policy documents. The system refuses to answer when the documents don't cover a topic, and flags conflicts between policies rather than guessing.

**What v1 won't do well:** It struggles with ambiguous synonyms — if an employee says "BYOD" and your policies say "personal device," it may miss the right document. It also won't catch conflicts between policies that don't happen to appear together in the same search result set.

## 7. Version 2 Roadmap

Version 1 intentionally prioritizes:

* retrieval correctness
* citation reliability
* operational simplicity
* low hallucination risk

The following improvements are recommended for Version 2.

---

### 7.1 Hybrid Retrieval (Keyword + Vector)

Current retrieval is purely semantic.

Problem:

* embeddings sometimes miss exact terminology
* acronyms like "BYOD" may fail retrieval

v2 improvement:

* combine BM25 keyword search + vector search
* fuse rankings using Reciprocal Rank Fusion (RRF)

Expected benefit:

* better recall on enterprise terminology
* fewer false negatives

---

### 7.2 Automated Policy Change Detection

Current ingestion assumes periodic refresh jobs.

Problem:

* stale chunks may survive after document updates

v2 improvement:

* webhook/event-driven ingestion
* automatic re-indexing on SharePoint/GDrive updates
* policy diff summaries for reviewers

Expected benefit:

* reduced stale-answer risk
* near-real-time policy freshness

---

### 7.3 Human Feedback Loop

Current evals are passive.

v2 improvement:

* employee thumbs-up/down feedback
* reviewer correction workflow
* low-confidence review queue

Expected benefit:

* continuously improving retrieval quality
* domain synonym discovery
* measurable trust metrics

---

### 7.4 Fine-Grained Access Control

Current design assumes all employees can access all policies.

v2 improvement:

* role-aware retrieval filtering
* department-level authorization
* confidential policy isolation

Expected benefit:

* enterprise security compliance
* safer deployment in regulated environments

---

### 7.5 Advanced Chunking Strategy

Current chunking uses fixed token windows.

Problem:

* sections may split mid-thought
* tables lose semantic structure

v2 improvement:

* semantic chunking
* heading-aware splitting
* table-aware parsing
* hierarchical retrieval

Expected benefit:

* improved citation precision
* better multi-section reasoning

---

### 7.6 Continuous Evaluation Dashboard

Current eval logging is database-only.

v2 improvement:

* real-time evaluation dashboard
* hallucination trend monitoring
* retrieval score analytics
* confidence drift alerts

Expected benefit:

* operational observability
* regression detection
* measurable SLA tracking

---

### 7.7 Multi-Turn Conversational Context

Current v1 treats every question independently.

v2 improvement:

* conversation memory
* follow-up question resolution
* contextual clarification

Example:

```text
User: Can I use my personal laptop?
User: What if I'm remote?
```

Expected benefit:

* more natural employee interaction
* reduced repeated queries

---

### 7.8 Enterprise Governance Features

Future enterprise deployments may require:

* audit trails
* legal hold support
* retention policies
* prompt versioning
* answer reproducibility
* signed policy snapshots

These are intentionally out-of-scope for v1 but important for regulated industries.

---

### 7.9 Cost and Latency Optimization

v1 optimizes for correctness over efficiency.

v2 optimization ideas:

* semantic caching
* adaptive retrieval depth
* smaller reranker for low-risk queries
* batch embedding generation
* response streaming

Expected benefit:

* lower operational cost
* faster employee response times

---

### 7.10 Multimodal Policy Understanding

Future policies may include:

* screenshots
* diagrams
* workflow charts
* scanned forms

v2 improvement:

* multimodal embeddings
* OCR pipelines
* image-aware retrieval

Expected benefit:

* broader policy coverage
* improved enterprise compatibility

```
```
## 1.5 Prerequisites / Operational Assumptions

The following infrastructure and organizational requirements are assumed for v1 deployment.

### Infrastructure Prerequisites

| Requirement                        | Purpose                                                                 |
| ---------------------------------- | ----------------------------------------------------------------------- |
| **Anthropic API Key**              | Required for Claude generation calls in the runtime query pipeline      |
| **PostgreSQL Instance**            | Stores chunk embeddings, metadata, and evaluation logs                  |
| **pgvector Extension Enabled**     | Enables vector similarity search inside PostgreSQL                      |
| **Centralized Policy Storage**     | Policies must live in a single SharePoint, Google Drive, or S3 location |
| **Python Runtime Environment**     | Python 3.10+ environment for ingestion and runtime services             |
| **Network Access to LLM Provider** | Runtime service must be able to reach Anthropic API endpoints           |
| **Policy Metadata Convention**     | Each document must have a stable policy ID and owner                    |

---

### Required Environment Variables

The runtime service expects the following environment variables:

```env
ANTHROPIC_API_KEY=<api_key>

PG_HOST=localhost
PG_PORT=5432
PG_DB=policydb
PG_USER=postgres
PG_PASSWORD=password
```

---

### Database Requirements

The PostgreSQL instance must:

* support the `pgvector` extension
* allow vector indexing
* provide persistent storage for:

  * policy chunks
  * embeddings
  * evaluation logs
  * ingestion metadata

Required extension:

```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

---

### Policy Document Requirements

For best retrieval quality, policies should:

* use consistent terminology
* contain section headings
* avoid scanned-image-only PDFs
* maintain version history
* include a document owner
* include effective dates

Poor document hygiene directly reduces retrieval quality.

---

### Operational Assumptions

This v1 system assumes:

1. Policies are primarily English-language text documents.
2. Policies change infrequently (daily re-ingestion is sufficient).
3. The company accepts "human escalation" for ambiguous/conflicting cases.
4. The assistant is advisory, not legally authoritative.
5. Employees understand that citations should be reviewed before actioning high-risk decisions.

---

### Security Assumptions

v1 assumes:

* internal-only deployment
* authenticated employee access
* no customer-facing exposure
* no PII redaction pipeline yet
* no tenant isolation requirements

These become mandatory in v2 for enterprise rollout.

```
```


---


