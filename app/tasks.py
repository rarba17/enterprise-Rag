from .celery_app import celery_app
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import PGVector
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

# DB Connection String for pgvector
CONNECTION_STRING = "postgresql+psycopg2://postgres:password@localhost:5432/rag_project"

@celery_app.task(bind=True, max_retries=3)
def process_document_task(self, file_path: str):
    try:
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
        return {"status": "success", "chunks_processed": len(chunks)}

    except Exception as exc:
        # Robustness: Retries every 60 seconds if OpenAI or DB fails
        raise self.retry(exc=exc, countdown=60)