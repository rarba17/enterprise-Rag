from sqlalchemy import Column, String, DateTime, JSON
from sqlalchemy.dialects.postgresql import UUID
import uuid
from datetime import datetime
from .database import Base

class Document(Base):
    __tablename__ = "documents"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    filename = Column(String, nullable=False)
    status = Column(String, default="PENDING") # PENDING, PROCESSING, COMPLETED, FAILED
    task_id = Column(String, nullable=True)     # The Celery Task ID
    created_at = Column(DateTime, default=datetime.utcnow)
    error_log = Column(String, nullable=True)   # Store error messages here if it fails