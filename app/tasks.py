import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, update
from sqlalchemy.orm import Session
from .celery_app import celery_app
from .models import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import PGVector
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

load_dotenv()

# DB Connection String for pgvector (sync driver for Celery)
CONNECTION_STRING = os.getenv(
    "SYNC_DATABASE_URL",
    "postgresql+psycopg2://postgres:password@localhost:5433/rag_db"
)

# Sync engine for Celery tasks
sync_engine = create_engine(CONNECTION_STRING)


def update_document_status(doc_id: str, status: str, error_log: str = None):
    """Update document status in database."""
    with Session(sync_engine) as session:
        stmt = update(Document).where(Document.id == doc_id).values(
            status=status,
            error_log=error_log
        )
        session.execute(stmt)
        session.commit()


@celery_app.task(bind=True, max_retries=3)
def process_document_task(self, file_path: str, doc_id: str):
    try:
        # Update status to PROCESSING
        update_document_status(doc_id, "PROCESSING")

        # 1. Load and Split PDF
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = text_splitter.split_documents(docs)

        # 2. Embed and Store in Postgres via pgvector
        embeddings = OpenAIEmbeddings()

        PGVector.from_documents(
            embedding=embeddings,
            documents=chunks,
            collection_name="enterprise_docs",
            connection_string=CONNECTION_STRING,
        )

        # Update status to COMPLETED
        update_document_status(doc_id, "COMPLETED")
        return {"status": "success", "chunks_processed": len(chunks)}

    except Exception as exc:
        # Update status to FAILED with error message
        update_document_status(doc_id, "FAILED", str(exc))
        # Robustness: Retries every 60 seconds if OpenAI or DB fails
        raise self.retry(exc=exc, countdown=60)