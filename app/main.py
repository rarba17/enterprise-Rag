# main.py
from fastapi import FastAPI, UploadFile, File, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from .database import get_db
from .models import Document
from .tasks import process_document_task
from langchain.globals import set_llm_cache
from langchain_community.cache import RedisSemanticCache
from contextlib import asynccontextmanager
from langchain_openai import OpenAIEmbeddings
import shutil
import os
import uuid
from sqlalchemy import text
from .database import engine,Base


@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- Startup Logic ---
    print("🛠️  Creating database tables...")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
    print("🚀 Initializing Semantic Cache...")
    set_llm_cache(RedisSemanticCache(
        redis_url="redis://localhost:6379/0",
        embedding=OpenAIEmbeddings()
    ))
    yield
    # --- Shutdown Logic (if needed) ---
    print("🛑 Shutting down...")

app = FastAPI()

@app.get("/")
async def root():
    return {"status": "Cache initialized and running"}

@app.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db)
):
    # 1. Save file to local storage
    os.makedirs("uploads", exist_ok=True)
    file_path = f"uploads/{uuid.uuid4()}_{file.filename}"
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # 2. Create entry in Postgres
    new_doc = Document(filename=file.filename, status="PENDING")
    db.add(new_doc)
    await db.commit()
    await db.refresh(new_doc)

    # 3. Trigger Event-Driven Task
    # We pass the ID and Path so the worker knows what to process
    task = process_document_task.delay(file_path, str(new_doc.id))

    # 4. Update the doc with the celery task_id
    new_doc.task_id = task.id
    await db.commit()

    return {"document_id": new_doc.id, "status": "Processing Started"}