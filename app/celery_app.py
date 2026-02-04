from celery import Celery
import os
from dotenv import load_dotenv

load_dotenv()

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6380/0")

celery_app = Celery(
    "worker",
    broker=REDIS_URL,
    backend=REDIS_URL,
    include=["app.tasks"]  # Explicitly include tasks module
)

celery_app.conf.task_routes = {
    "app.tasks.process_document_task": "main-queue"
}

celery_app.conf.broker_connection_retry_on_startup = True