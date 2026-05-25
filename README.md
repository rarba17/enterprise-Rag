# Enterprise RAG - Complete Technical Architecture & System Documentation

> **Version:** 1.0.0
> **System Type:** Production-Ready, Event-Driven Retrieval-Augmented Generation Backend
> **Primary Language:** Python 3.11+
> **Core Frameworks:** FastAPI, LangChain, Celery, SQLAlchemy, pgvector

---

## Table of Contents

1. [High-Level System Overview](#1-high-level-system-overview)
2. [Repository Structure Breakdown](#2-repository-structure-breakdown)
3. [Architecture Diagram (Textual)](#3-architecture-diagram-textual)
4. [Core Business Logic Explanation](#4-core-business-logic-explanation)
5. [Agentic AI Layer Deep Dive](#5-agentic-ai-layer-deep-dive)
6. [API Layer Analysis](#6-api-layer-analysis)
7. [End-to-End Data Flow](#7-end-to-end-data-flow)
8. [Execution & Runtime Flow](#8-execution--runtime-flow)
9. [Onboarding Section](#9-onboarding-section)
10. [Strengths & Design Trade-offs](#10-strengths--design-trade-offs)

---

## 1. High-Level System Overview

### What It Does

Enterprise RAG is a **production-grade backend system** that allows organizations to upload PDF documents, automatically process them into searchable vector embeddings, and then ask natural-language questions that are answered by an LLM grounded in the content of those documents. This is the core pattern known as **Retrieval-Augmented Generation (RAG)** — the LLM does not hallucinate from general knowledge; it retrieves actual passages from your documents and synthesizes a precise answer.

### Who It Is For

- **Enterprises** needing internal document Q&A (compliance docs, research papers, HR policies)
- **Knowledge teams** that want to query across a corpus of PDFs without reading every page
- **Developers** building AI-powered search into existing products

### The Problem It Solves

Raw LLMs cannot answer questions about private, domain-specific documents. Enterprise RAG bridges this gap by:

1. **Ingesting** PDFs asynchronously (no blocking the user)
2. **Chunking** documents into semantically meaningful segments
3. **Embedding** those chunks into high-dimensional vectors stored in PostgreSQL (pgvector)
4. **Retrieving** the most relevant chunks for any given question via Maximal Marginal Relevance (MMR) search
5. **Generating** a grounded answer using GPT-4o with only the retrieved context

### Technology Stack Summary

| Layer | Technology | Purpose |
|---|---|---|
| API Framework | FastAPI (async) | High-performance REST + WebSocket API |
| AI Orchestration | LangChain + OpenAI GPT-4o | RAG chain construction, embeddings, LLM calls |
| Vector Database | PostgreSQL + pgvector | Stores and searches document embeddings |
| Task Queue | Celery + Redis | Async document processing (event-driven) |
| Caching Layer | Redis (semantic cache ready) | Reduces redundant LLM calls for similar queries |
| Authentication | JWT (HS256) + bcrypt | Stateless token-based auth with password hashing |
| ORM | SQLAlchemy 2.0 (async) | Async database access with connection pooling |
| Frontend | Streamlit | Chat-style UI with WebSocket streaming |
| Containerization | Docker Compose | PostgreSQL (pgvector) + Redis infrastructure |

---

## 2. Repository Structure Breakdown

```
enterprise-rag/
├── app/                          # Core application package
│   ├── main.py                   # FastAPI app factory, lifespan, router registration
│   ├── database.py               # Async SQLAlchemy engine, session factory, DB dependency
│   ├── models.py                 # SQLAlchemy ORM models (User, Document)
│   ├── schemas.py                # Pydantic request/response schemas
│   ├── security.py               # JWT creation/validation, bcrypt hashing, auth dependencies
│   ├── query.py                  # RAG engine: vector retrieval, LLM chain, streaming
│   ├── tasks.py                  # Celery task: PDF loading, chunking, embedding
│   ├── celery_app.py             # Celery broker/backend configuration
│   └── routes/                   # API route modules
│       ├── __init__.py           # Router barrel exports
│       ├── auth.py               # /auth/* endpoints (register, login, me)
│       ├── documents.py          # /documents/* endpoints (upload, list, get, delete)
│       └── query.py              # /query/* endpoints (ask, debug, WebSocket stream)
│
├── streamlit_app.py              # Full-featured Streamlit frontend (chat UI)
├── docker-compose.yml            # Infrastructure: pgvector PostgreSQL + Redis
├── requirements.txt              # Pinned Python dependencies
├── .env                          # Environment variables (secrets, connection strings)
├── .gitignore                    # Excludes .env, uploads/, __pycache__, etc.
├── test.py                       # End-to-end pipeline test (upload -> query -> cache)
├── test_auth.py                  # Authentication flow integration test
├── uploads/                      # Local PDF storage (gitignored)
└── README.md                     # Quick-start instructions
```

### Directory-by-Directory Explanation

**`app/`** — The core backend. Every module has a single responsibility. The `routes/` sub-package isolates HTTP concerns from business logic. The `query.py` module is the AI brain. The `tasks.py` module is the async worker brain.

**`uploads/`** — Transient file storage. PDFs land here after upload, are read by the Celery worker during processing, and remain as a reference. Gitignored to avoid committing user data.

**Root files** — `docker-compose.yml` provisions the infrastructure. `streamlit_app.py` is a standalone frontend. Test scripts validate the full flow end-to-end.

---

## 3. Architecture Diagram (Textual)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          CLIENT LAYER                                   │
│                                                                         │
│   ┌──────────────┐    ┌──────────────────┐    ┌──────────────────┐     │
│   │  Streamlit   │    │  API Consumers   │    │  WebSocket       │     │
│   │  Frontend    │    │  (curl/Postman)  │    │  Clients         │     │
│   └──────┬───────┘    └────────┬─────────┘    └────────┬─────────┘     │
│          │ HTTP/REST           │ HTTP/REST              │ WS            │
└──────────┼─────────────────────┼───────────────────────┼───────────────┘
           │                     │                       │
           ▼                     ▼                       ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                       FastAPI APPLICATION (async)                        │
│                                                                         │
│   ┌────────────────┐  ┌────────────────┐  ┌────────────────────────┐   │
│   │  Auth Router   │  │  Documents     │  │  Query Router          │   │
│   │  /auth/*       │  │  Router        │  │  /query/ask (REST)     │   │
│   │  register      │  │  /documents/*  │  │  /query/debug (REST)   │   │
│   │  login         │  │  upload        │  │  /query/ws (WebSocket) │   │
│   │  me            │  │  list/get/del  │  │                        │   │
│   └───────┬────────┘  └───────┬────────┘  └───────┬────────────────┘   │
│           │                   │                    │                     │
│   ┌───────▼────────┐         │            ┌───────▼────────────────┐   │
│   │  Security      │         │            │  RAG Engine            │   │
│   │  Module        │         │            │  (query.py)            │   │
│   │  JWT + bcrypt  │         │            │  LangChain + GPT-4o   │   │
│   └───────┬────────┘         │            │  MMR Retrieval         │   │
│           │                  │            │  Streaming Support     │   │
│           ▼                  ▼            └───────┬────────────────┘   │
│   ┌──────────────────────────────────┐           │                     │
│   │       SQLAlchemy Async ORM       │           │                     │
│   │       (asyncpg driver)           │           │                     │
│   └──────────────┬───────────────────┘           │                     │
└──────────────────┼───────────────────────────────┼─────────────────────┘
                   │                               │
                   │                               │ LangChain pgvector
                   │                               │ (psycopg2 sync driver)
                   ▼                               ▼
┌─────────────────────────────────────────────────────────────────────────┐
│              PostgreSQL 16 + pgvector Extension                          │
│                                                                         │
│   ┌────────────────────┐       ┌─────────────────────────────────┐     │
│   │  users table       │       │  langchain_pg_embedding         │     │
│   │  documents table   │       │  langchain_pg_collection        │     │
│   │  (relational)      │       │  (vector embeddings, 1536-dim)  │     │
│   └────────────────────┘       └─────────────────────────────────┘     │
└─────────────────────────────────────────────────────────────────────────┘

                   ASYNC DOCUMENT PROCESSING PIPELINE

┌──────────────┐    ┌───────────────┐    ┌──────────────────────────┐
│  FastAPI      │    │    Redis      │    │   Celery Worker          │
│  (producer)   │───▶│  (broker +    │───▶│   process_document_task  │
│  upload route │    │   backend)    │    │                          │
│  .delay()     │    │  main-queue   │    │  1. Load PDF (PyPDF)     │
└──────────────┘    └───────────────┘    │  2. Chunk (500 chars)    │
                                          │  3. Embed (OpenAI)       │
                            ┌─────────────│  4. Store (pgvector)     │
                            │             │  5. Update status in DB  │
                            ▼             └──────────────────────────┘
                    ┌───────────────┐
                    │  PostgreSQL   │
                    │  (status +    │
                    │   vectors)    │
                    └───────────────┘
```

### Component Interaction Summary

| From | To | Protocol | Purpose |
|---|---|---|---|
| Client | FastAPI | HTTP/REST | Auth, document CRUD, queries |
| Client | FastAPI | WebSocket | Real-time streaming answers |
| FastAPI | PostgreSQL | asyncpg (async) | User/document CRUD |
| FastAPI | Redis | Celery `.delay()` | Enqueue document processing |
| Celery Worker | PostgreSQL | psycopg2 (sync) | Status updates + vector storage |
| Celery Worker | OpenAI API | HTTPS | Generate embeddings |
| Query Engine | PostgreSQL/pgvector | psycopg2 (sync) | Similarity search |
| Query Engine | OpenAI API | HTTPS | LLM answer generation |
| Redis | Celery Worker | AMQP-like | Task dispatch + result backend |

---

## 4. Core Business Logic Explanation

The system's business logic can be decomposed into three pillars:

### Pillar 1: User Authentication & Authorization

**Module:** `app/security.py` + `app/routes/auth.py`

- **Registration:** Email + password. Password is hashed with bcrypt (salt auto-generated). Duplicate email check via async DB query. User record stored with UUID primary key.
- **Login:** Accepts `OAuth2PasswordRequestForm` (standard OAuth2 form fields). Verifies password against stored bcrypt hash. Issues a JWT (HS256) containing `{"sub": user_id}` with a configurable expiration (default 30 minutes).
- **Route Protection:** All `/documents/*` and `/query/*` endpoints depend on `get_current_active_user`, which:
  1. Extracts the JWT from the `Authorization: Bearer <token>` header
  2. Decodes and validates the token (signature + expiry)
  3. Looks up the user by ID from the token payload
  4. Rejects inactive users with HTTP 403
- **Ownership Isolation:** Document queries are scoped by `owner_id`, ensuring users can only access their own documents.

### Pillar 2: Asynchronous Document Ingestion

**Module:** `app/tasks.py` + `app/routes/documents.py`

The upload flow is a **producer-consumer pattern**:

1. **Producer (FastAPI):** Receives PDF upload, validates file type and size (max 10MB), saves to disk with UUID-prefixed filename, creates a `Document` record with status `PENDING`, dispatches a Celery task via `.delay()`, and returns immediately with the document ID.

2. **Consumer (Celery Worker):** Picks up the task from the Redis `main-queue`, transitions status to `PROCESSING`, then executes the RAG ingestion pipeline:
   - **Load:** `PyPDFLoader` extracts text from each PDF page
   - **Chunk:** `RecursiveCharacterTextSplitter` breaks text into 500-character chunks with 100-character overlap, using a hierarchical separator strategy (`\n\n` > `\n` > `. ` > ` ` > `""`)
   - **Embed:** `OpenAIEmbeddings` (text-embedding-ada-002) converts each chunk to a 1536-dimensional vector
   - **Store:** `PGVector.from_documents` writes embeddings to the `langchain_pg_embedding` table under collection `enterprise_docs`
   - **Finalize:** Status transitions to `COMPLETED` (or `FAILED` with error log)

3. **Retry Policy:** On failure, the task retries up to 3 times with a 60-second backoff (`max_retries=3, countdown=60`).

### Pillar 3: RAG Query Engine

**Module:** `app/query.py`

Two query modes exist:

**Synchronous (REST):**
1. Instantiate `PGVector` from the existing `enterprise_docs` collection
2. Create an MMR retriever (`k=8, fetch_k=20`) — fetches 20 candidates, selects 8 that maximize both relevance and diversity
3. Build a `RetrievalQA` chain (LangChain) with a custom prompt template that instructs the LLM to answer only from context
4. Invoke GPT-4o with `temperature=0` for deterministic, factual answers
5. Return the answer with source documents

**Streaming (WebSocket):**
1. Same retrieval step (MMR, k=8)
2. Manually assembles context from retrieved documents
3. Uses `ChatOpenAI.astream()` to produce token-by-token output
4. Each token is pushed to the WebSocket client as `{"chunk": "..."}`, with `{"done": true}` signaling completion

### Prompt Engineering

The prompt template is deliberately constrained:

```
Use the following pieces of context to answer the question.
If you find the answer in the context, provide it directly and precisely.
If the answer is not in the context, say "I couldn't find this information in the documents."
```

This **grounding instruction** prevents hallucination — the LLM will refuse to answer rather than fabricate information not present in the retrieved chunks.

---

## 5. Agentic AI Layer Deep Dive

### How the RAG "Agent" Operates

While this system does not use an explicit multi-agent orchestration framework (e.g., LangGraph, CrewAI), it implements a **single-agent RAG architecture** where the AI layer makes autonomous retrieval and synthesis decisions:

#### Agent Initialization
- The "agent" is instantiated per-request in `query.py` — there is no persistent agent state between requests
- Each invocation creates a fresh `ChatOpenAI` (GPT-4o) instance and a fresh `PGVector` retriever
- This is a **stateless agent design** — advantageous for horizontal scaling

#### Decision Flow
```
User Question
     │
     ▼
┌─────────────┐
│  Retriever   │  MMR algorithm decides which chunks are most relevant
│  (pgvector)  │  AND most diverse (avoids redundant context)
└──────┬──────┘
       │ Top 8 chunks
       ▼
┌─────────────┐
│  Prompt      │  Injects retrieved context into a structured prompt
│  Assembly    │  with explicit grounding instructions
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  LLM        │  GPT-4o synthesizes answer from context only
│  (GPT-4o)   │  temperature=0 for deterministic output
└──────┬──────┘
       │
       ▼
   Final Answer
```

#### Retrieval Strategy: Why MMR?

The retriever uses **Maximal Marginal Relevance** instead of pure cosine similarity. This is a deliberate architectural choice:

- **Pure similarity search** can return 8 chunks that all say roughly the same thing (high redundancy)
- **MMR** balances relevance to the query with diversity among selected chunks, ensuring the LLM sees a broader coverage of the document's content
- Parameters: `fetch_k=20` retrieves 20 candidates, then MMR selects the 8 most informative and diverse

#### Chunking Strategy: Why 500/100?

- **chunk_size=500** characters captures granular facts (names, numbers, specific claims) without losing too much context
- **chunk_overlap=100** ensures that sentences split across chunk boundaries are still captured in at least one chunk
- The hierarchical separator list (`\n\n`, `\n`, `. `, ` `, `""`) tries to split at natural paragraph/sentence boundaries first, falling back to word and character boundaries only as a last resort

#### Memory / State Handling

- **No conversational memory** — each query is independent. The system does not maintain chat history on the server side. This is by design for a document Q&A system (not a chatbot).
- **Semantic cache (disabled but wired):** `main.py` contains commented-out `RedisSemanticCache` configuration. When enabled, it hashes the semantic embedding of each query and returns cached LLM responses for sufficiently similar questions (`score_threshold=0.05`). This would dramatically reduce OpenAI API costs for repeated/similar queries.

#### External Tool Calls

The agent's only "tool" is the vector store retriever. It does not call external APIs, execute code, or perform web searches. This is a focused **retrieve-then-generate** agent, not a general-purpose tool-using agent.

---

## 6. API Layer Analysis

### Endpoint Reference

#### Authentication (`/auth`)

| Method | Path | Auth | Request | Response | Status |
|---|---|---|---|---|---|
| POST | `/auth/register` | None | `{"email": "...", "password": "..."}` | `UserResponse` (id, email, is_active, created_at) | 201 |
| POST | `/auth/login` | None | OAuth2 form: `username` + `password` | `{"access_token": "...", "token_type": "bearer"}` | 200 |
| GET | `/auth/me` | Bearer JWT | — | `UserResponse` | 200 |

#### Documents (`/documents`)

| Method | Path | Auth | Request | Response | Status |
|---|---|---|---|---|---|
| POST | `/documents/upload` | Bearer JWT | Multipart file upload (PDF, max 10MB) | `DocumentResponse` (id, filename, status, task_id, created_at, owner_id) | 200 |
| GET | `/documents` | Bearer JWT | — | `List[DocumentResponse]` | 200 |
| GET | `/documents/{id}` | Bearer JWT | — | `DocumentResponse` | 200 |
| DELETE | `/documents/{id}` | Bearer JWT | — | — | 204 |

#### Query (`/query`)

| Method | Path | Auth | Request | Response | Status |
|---|---|---|---|---|---|
| GET | `/query/ask` | Bearer JWT | Query param: `?question=...` | `{"question": "...", "answer": "..."}` | 200 |
| GET | `/query/debug` | Bearer JWT | Query param: `?question=...` | List of chunks with scores and metadata | 200 |
| WS | `/query/ws` | JWT via query param `?token=...` | JSON: `{"question": "..."}` | Stream: `{"chunk": "..."}` ... `{"done": true}` | — |

#### Health

| Method | Path | Auth | Response |
|---|---|---|---|
| GET | `/` | None | `{"status": "healthy", "service": "...", "version": "1.0.0"}` |
| GET | `/health` | None | `{"status": "healthy", "database": "connected", ...}` |

### Request/Response Lifecycle

```
HTTP Request
     │
     ▼
FastAPI Middleware (ASGI)
     │
     ▼
Router Matching (/auth, /documents, /query)
     │
     ▼
Dependency Injection:
  ├── get_db() → AsyncSession (scoped per request)
  ├── oauth2_scheme → extracts Bearer token
  └── get_current_active_user → validates JWT, loads User from DB
     │
     ▼
Route Handler (business logic)
     │
     ▼
Pydantic Response Model Serialization
     │
     ▼
HTTP Response (JSON)
```

### Validation

- **Email validation:** `pydantic.EmailStr` enforces valid email format at registration
- **File type validation:** Upload endpoint checks `.pdf` extension
- **File size validation:** 10MB hard limit with seek-based size check (no full read into memory)
- **JWT validation:** Signature verification, expiry check, user existence check, active status check
- **Ownership validation:** Document queries filter by `owner_id` — users cannot access other users' documents

---

## 7. End-to-End Data Flow

### Flow 1: Document Upload → Processing → Queryable

```
                    TIME ──────────────────────────────────────────────▶

User              FastAPI                  Redis          Celery Worker         PostgreSQL         OpenAI
 │                   │                      │                  │                    │                │
 │── POST /upload ──▶│                      │                  │                    │                │
 │   (PDF file)      │                      │                  │                    │                │
 │                   │── save to disk ──────────────────────────────────────────────▶│                │
 │                   │── INSERT Document ───────────────────────────────────────────▶│ (PENDING)      │
 │                   │── task.delay() ─────▶│                  │                    │                │
 │◀── 200 {doc_id} ─│                      │                  │                    │                │
 │                   │                      │── dispatch ─────▶│                    │                │
 │                   │                      │                  │── UPDATE status ──▶│ (PROCESSING)   │
 │                   │                      │                  │── load PDF ────────│                │
 │                   │                      │                  │── chunk text ──────│                │
 │                   │                      │                  │── embed chunks ────│───────────────▶│
 │                   │                      │                  │◀──── vectors ──────│───────────────◀│
 │                   │                      │                  │── store vectors ──▶│                │
 │                   │                      │                  │── UPDATE status ──▶│ (COMPLETED)    │
 │                   │                      │                  │                    │                │
```

### Flow 2: Question → Retrieval → Answer

```
User              FastAPI              pgvector (PostgreSQL)      OpenAI
 │                   │                        │                     │
 │── GET /ask?q= ──▶│                        │                     │
 │                   │── embed question ──────│────────────────────▶│
 │                   │◀── question vector ────│────────────────────◀│
 │                   │── MMR search (k=8) ───▶│                     │
 │                   │◀── top 8 chunks ───────│                     │
 │                   │── prompt + context ────│────────────────────▶│
 │                   │◀── generated answer ───│────────────────────◀│
 │◀── 200 {answer} ─│                        │                     │
```

### Flow 3: WebSocket Streaming Query

```
User (WS)         FastAPI (WS)          pgvector              OpenAI (streaming)
 │                    │                     │                      │
 │── connect ────────▶│                     │                      │
 │◀── accept ─────────│                     │                      │
 │── {"question":} ──▶│                     │                      │
 │                    │── MMR search ───────▶│                      │
 │                    │◀── chunks ───────────│                      │
 │                    │── stream request ────│─────────────────────▶│
 │◀── {"chunk":"Th"} ─│◀── token ───────────│─────────────────────◀│
 │◀── {"chunk":"e "}  │◀── token            │                      │
 │◀── {"chunk":"ans"} │◀── token            │                      │
 │◀── ...             │◀── ...              │                      │
 │◀── {"done":true} ──│                     │                      │
```

---

## 8. Execution & Runtime Flow

### Startup Sequence

The system requires **four processes** running concurrently:

```
Process 1: Docker Compose (infrastructure)
  $ docker-compose up -d
  ├── pgvector/pgvector:pg16  →  port 5433 (host) → 5432 (container)
  └── redis-stack-server      →  port 6380 (host) → 6379 (container)

Process 2: FastAPI Application Server
  $ uvicorn app.main:app --reload
  ├── Triggers lifespan handler:
  │   ├── CREATE ALL tables (users, documents)
  │   ├── CREATE EXTENSION vector (enables pgvector)
  │   └── (optional) Initialize Redis semantic cache
  ├── Registers routers: /auth, /documents, /query
  └── Listens on port 8000

Process 3: Celery Worker
  $ celery -A app.celery_app worker --loglevel=info -Q main-queue
  ├── Connects to Redis broker (port 6380)
  ├── Subscribes to "main-queue"
  └── Waits for process_document_task dispatches

Process 4: Streamlit Frontend (optional)
  $ streamlit run streamlit_app.py
  ├── Connects to FastAPI at http://localhost:8000
  └── Serves UI on port 8501
```

### Request Lifecycle (Detailed)

1. **ASGI receives request** — Uvicorn passes to FastAPI
2. **Route matching** — FastAPI matches URL pattern to router
3. **Dependency resolution** — FastAPI's DI container resolves:
   - `get_db()` opens an `AsyncSession` from the connection pool (pool_size=20, max_overflow=40)
   - `oauth2_scheme` extracts the Bearer token from the Authorization header
   - `get_current_user` decodes JWT, queries DB for user
4. **Handler execution** — Business logic runs (e.g., document upload, RAG query)
5. **Response serialization** — Pydantic model validates and serializes the response
6. **Session cleanup** — `get_db()` context manager closes the session
7. **Response sent** — JSON response returned to client

### Database Connection Strategy

The system uses **two separate database drivers** — this is a deliberate and necessary design decision:

| Driver | Used By | Why |
|---|---|---|
| `asyncpg` (async) | FastAPI routes, SQLAlchemy ORM | Non-blocking I/O for high-concurrency HTTP handling |
| `psycopg2` (sync) | Celery tasks, LangChain/pgvector | Celery workers are synchronous; LangChain's pgvector integration requires a sync driver |

Both drivers connect to the **same PostgreSQL instance** but via different connection strings configured in `.env`.

---

## 9. Onboarding Section

### Prerequisites

- Python 3.11+
- Docker and Docker Compose
- An OpenAI API key

### Step-by-Step Setup

```bash
# 1. Clone the repository
git clone <repo-url> && cd enterprise-rag

# 2. Create and activate virtual environment
python -m venv .venv && source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment variables
# Copy and edit the .env file (set your own OPENAI_API_KEY and JWT_SECRET_KEY)
# Generate a JWT secret: openssl rand -hex 32

# 5. Start infrastructure
docker-compose up -d

# 6. Start the Celery worker (in a separate terminal)
celery -A app.celery_app worker --loglevel=info -Q main-queue

# 7. Start the FastAPI server (in a separate terminal)
uvicorn app.main:app --reload

# 8. (Optional) Start the Streamlit frontend
streamlit run streamlit_app.py
```

### Verify It Works

```bash
# Health check
curl http://localhost:8000/

# Run the auth integration test
python test_auth.py
```

### How to Navigate the Codebase

**Start here** — read these files in this order to understand the system:

1. **`app/main.py`** — Entry point. See what routers exist and how the app boots.
2. **`app/models.py`** — Domain models. Understand the data shape.
3. **`app/routes/auth.py`** — Auth flow. Simple and self-contained.
4. **`app/routes/documents.py`** — Upload flow. See how Celery tasks are dispatched.
5. **`app/tasks.py`** — The RAG ingestion pipeline. This is where AI magic begins.
6. **`app/query.py`** — The RAG query engine. This is where AI magic completes.
7. **`app/routes/query.py`** — How queries are exposed via REST and WebSocket.
8. **`app/security.py`** — JWT and auth internals.
9. **`streamlit_app.py`** — Frontend. Standalone; good for understanding the user experience.

### Key Concepts a New Developer Must Understand

1. **FastAPI dependency injection** — `Depends()` is used everywhere for DB sessions and auth. Understand how `get_db()` and `get_current_active_user` work.
2. **Celery async tasks** — `.delay()` enqueues a task; it runs in a separate process. The FastAPI app and Celery worker share the same codebase but run independently.
3. **LangChain chains** — `RetrievalQA.from_chain_type()` wires together a retriever, prompt, and LLM into a callable pipeline.
4. **pgvector** — PostgreSQL extension that adds vector column types and similarity search operators. LangChain abstracts this via `PGVector`.
5. **Two DB drivers** — The async/sync split is intentional. Don't try to use `asyncpg` in Celery tasks or `psycopg2` in FastAPI routes.

---

## 10. Strengths & Design Trade-offs

### Architectural Strengths

#### 1. Event-Driven Document Processing
The Celery-based async pipeline means the API never blocks on heavy operations (PDF parsing, embedding generation). Users get an immediate response and can poll for status. This is essential for production systems where PDF processing can take 10-60 seconds.

#### 2. Dual-Protocol Query Interface
Supporting both REST (simple, cacheable) and WebSocket (real-time streaming) gives consumers flexibility. The streaming path dramatically improves perceived latency — users see tokens arriving in real-time rather than waiting for the full response.

#### 3. MMR Retrieval Over Naive Similarity
Most RAG tutorials use simple cosine similarity. This system uses MMR, which produces materially better answers by ensuring the LLM sees diverse context rather than redundant near-duplicates.

#### 4. Grounded Prompt Design
The prompt explicitly instructs the LLM to refuse answering when context is insufficient. This is a critical production concern — enterprise users need to trust that answers come from their documents, not from the LLM's training data.

#### 5. Production-Ready Connection Pooling
The async engine is configured with `pool_size=20, max_overflow=40, pool_pre_ping=True, pool_recycle=3600`. This handles connection storms, stale connections, and connection limits gracefully.

#### 6. Retry Semantics on Ingestion
The Celery task retries up to 3 times with 60-second backoff. Transient OpenAI API failures or database blips won't permanently fail a document ingestion.

#### 7. Ownership-Scoped Data Access
Documents are scoped to their owner via `owner_id` foreign key, and all queries filter by the authenticated user. This is a baseline multi-tenancy model.

### Design Trade-offs & Areas for Enhancement

#### 1. No Per-User Vector Isolation
While document records are scoped by `owner_id`, the vector store uses a single `enterprise_docs` collection for all users. This means RAG queries retrieve chunks from **all users' documents**, not just the querying user's. For true multi-tenancy, either:
- Filter vectors by owner metadata at query time
- Use per-user collections
- Add metadata filtering to the retriever

#### 2. Synchronous RAG Query in Async Route
`ask_question()` in `query.py` calls LangChain synchronously (it uses the sync psycopg2 driver). When called from an async FastAPI route, this blocks the event loop. For production, this should either:
- Run in a thread pool (`asyncio.to_thread()`)
- Use fully async LangChain integrations

#### 3. Semantic Cache Disabled
The Redis semantic cache is wired but commented out. When enabled, it would cache LLM responses for semantically similar queries, reducing OpenAI API costs significantly. The `score_threshold=0.05` (very strict) was chosen to avoid serving wrong cached answers.

#### 4. No Document Deletion from Vector Store
The `DELETE /documents/{id}` endpoint removes the document record from PostgreSQL but does not remove the corresponding vector embeddings from pgvector. Deleted documents' chunks remain queryable.

#### 5. File Storage is Local
Uploaded PDFs are stored on local disk (`uploads/`). For production deployment across multiple instances, this should be replaced with object storage (S3, GCS, MinIO).

#### 6. No Rate Limiting
There is no rate limiting on the OpenAI-calling endpoints (`/query/ask`, `/query/ws`). Each query costs real money. Adding rate limiting (e.g., via `slowapi` or Redis-based token buckets) would be important for production.

#### 7. Health Check is Static
The `/health` endpoint returns hardcoded "connected" values. A production health check should actually verify connectivity to PostgreSQL and Redis.

---

## Appendix A: Environment Variables

| Variable | Required | Default | Purpose |
|---|---|---|---|
| `OPENAI_API_KEY` | Yes | — | OpenAI API authentication |
| `DATABASE_URL` | Yes | `postgresql+asyncpg://...` | Async PostgreSQL connection (FastAPI) |
| `SYNC_DATABASE_URL` | Yes | `postgresql+psycopg2://...` | Sync PostgreSQL connection (Celery/LangChain) |
| `REDIS_URL` | Yes | `redis://localhost:6380/0` | Redis broker/backend for Celery |
| `JWT_SECRET_KEY` | Yes | — | HMAC secret for JWT signing (HS256) |
| `ACCESS_TOKEN_EXPIRE_MINUTES` | No | `30` | JWT token lifetime |

## Appendix B: Database Schema

### `users` Table

| Column | Type | Constraints |
|---|---|---|
| id | UUID | PK, default uuid4 |
| email | VARCHAR | UNIQUE, NOT NULL, INDEXED |
| hashed_password | VARCHAR | NOT NULL |
| is_active | BOOLEAN | default TRUE |
| created_at | TIMESTAMP WITH TZ | default now() |

### `documents` Table

| Column | Type | Constraints |
|---|---|---|
| id | UUID | PK, default uuid4, INDEXED |
| filename | VARCHAR | NOT NULL |
| status | VARCHAR | default "PENDING", INDEXED |
| task_id | VARCHAR | NULLABLE (Celery task ID) |
| created_at | TIMESTAMP WITH TZ | default now(), INDEXED |
| error_log | VARCHAR | NULLABLE |
| owner_id | UUID | FK → users.id, NULLABLE, INDEXED |

### `langchain_pg_collection` Table (managed by LangChain)

| Column | Type | Purpose |
|---|---|---|
| uuid | UUID | PK |
| name | VARCHAR | Collection name (`enterprise_docs`) |
| cmetadata | JSON | Collection metadata |

### `langchain_pg_embedding` Table (managed by LangChain)

| Column | Type | Purpose |
|---|---|---|
| uuid | UUID | PK |
| collection_id | UUID | FK → collection |
| embedding | VECTOR(1536) | OpenAI embedding vector |
| document | TEXT | Original chunk text |
| cmetadata | JSON | Source metadata (page, source file) |

---

*This document was generated through comprehensive static analysis of the Enterprise RAG codebase. All architectural claims are derived directly from source code, not assumptions.*
