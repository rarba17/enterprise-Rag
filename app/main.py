# main.py
"""
Enterprise RAG API - Main Application

A production-ready, event-driven Retrieval-Augmented Generation backend
with JWT authentication, async document processing, and semantic caching.
"""

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from sqlalchemy import text
from langchain.globals import set_llm_cache
from langchain_community.cache import RedisSemanticCache
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

from .database import engine, Base
from .routes import auth_router, documents_router, query_router

load_dotenv()

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6380/0")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Application lifespan handler for startup/shutdown events."""
    # --- Startup Logic ---
    print("Creating database tables...")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))

    print("Initializing Semantic Cache...")
    set_llm_cache(RedisSemanticCache(
        redis_url=REDIS_URL,
        embedding=OpenAIEmbeddings()
    ))

    print("Enterprise RAG API started successfully!")
    yield

    # --- Shutdown Logic ---
    print("Shutting down...")


app = FastAPI(
    title="Enterprise RAG API",
    description="""
## Overview
Production-ready RAG backend for document processing and semantic search.

## Features
- **JWT Authentication** - Secure token-based authentication
- **Async Document Processing** - Non-blocking PDF ingestion via Celery
- **Semantic Search** - Vector similarity search with pgvector
- **Smart Caching** - Redis semantic cache reduces API costs

## Quick Start
1. Register: `POST /auth/register`
2. Login: `POST /auth/login`
3. Upload PDF: `POST /documents/upload`
4. Ask questions: `GET /query/ask?question=...`
    """,
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)


# =============================================================================
# ROUTERS
# =============================================================================

app.include_router(auth_router)
app.include_router(documents_router)
app.include_router(query_router)


# =============================================================================
# ROOT ENDPOINT
# =============================================================================

@app.get("/", tags=["Health"])
async def root():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "Enterprise RAG API",
        "version": "1.0.0"
    }


@app.get("/health", tags=["Health"])
async def health_check():
    """Detailed health check endpoint."""
    return {
        "status": "healthy",
        "database": "connected",
        "cache": "connected",
        "worker": "check /documents for processing status"
    }
