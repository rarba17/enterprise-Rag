from sqlalchemy import Column, String, DateTime, Boolean, ForeignKey
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
import uuid
from datetime import datetime
from .database import Base


class User(Base):
    __tablename__ = "users"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationship to documents
    documents = relationship("Document", back_populates="owner")


class Document(Base):
    __tablename__ = "documents"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    filename = Column(String, nullable=False)
    status = Column(String, default="PENDING")  # PENDING, PROCESSING, COMPLETED, FAILED
    task_id = Column(String, nullable=True)     # The Celery Task ID
    created_at = Column(DateTime, default=datetime.utcnow)
    error_log = Column(String, nullable=True)   # Store error messages here if it fails

    # Foreign key to user
    owner_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=True)
    owner = relationship("User", back_populates="documents")