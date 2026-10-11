#!/usr/bin/env python3
"""Ingest Phase-2 public contexts (SQuAD v2) into the SAME pgvector collection.

Reuses YOUR production settings from app/tasks.py so public results are
comparable to synthetic: chunk 500/100, same separators, OpenAIEmbeddings,
collection 'enterprise_docs'. Tags metadata source=public for later cleanup.

Cost: ~10k tokens (~cents). Idempotent-ish: re-running re-embeds (pgvector
appends); to reset, delete rows with cmetadata->>'source'='public'.

Usage:
    .venv/bin/python eval/ingest_public.py [--limit 10]   # smoke test first
    .venv/bin/python eval/ingest_public.py                # full 78 contexts
"""
import argparse
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CTX = HERE / "public" / "contexts.json"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()

    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    from langchain_openai import OpenAIEmbeddings
    from langchain_community.vectorstores.pgvector import PGVector
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_core.documents import Document

    contexts = json.loads(CTX.read_text())
    if a.limit:
        contexts = contexts[:a.limit]
    print(f"ingesting {len(contexts)} public contexts...")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=100,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    docs: list[Document] = []
    for c in contexts:
        for chunk in splitter.split_text(c["context"]):
            docs.append(Document(
                page_content=chunk,
                metadata={"doc_id": c["doc_id"], "source": "public",
                          "title": c.get("title", "")[:120]}))
    print(f"  {len(docs)} chunks -> embedding (OPENAI_API_KEY from .env)...")

    conn = os.getenv("SYNC_DATABASE_URL",
                     "postgresql+psycopg2://postgres:password@localhost:5433/rag_db")
    PGVector.from_documents(
        embedding=OpenAIEmbeddings(),
        documents=docs,
        collection_name="enterprise_docs",
        connection_string=conn,
    )
    print(f"done. {len(docs)} public chunks stored (source=public).")
    print("verify: .venv/bin/python eval/eval_retrieval.py --dataset public --limit 5")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
