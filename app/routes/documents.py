"""Document management routes for upload and retrieval."""

import os
import shutil
import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..database import get_db
from ..models import Document, User
from ..schemas import DocumentResponse
from ..tasks import process_document_task
from ..security import get_current_active_user

router = APIRouter(prefix="/documents", tags=["Documents"])

UPLOAD_DIR = "uploads"


@router.post("/upload", response_model=DocumentResponse)
async def upload_document(
    current_user: Annotated[User, Depends(get_current_active_user)],
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db)
):
    """
    Upload a document for processing.

    - Accepts PDF files
    - Document is queued for async processing (chunking + embedding)
    - Returns document ID to track processing status
    """
    # Validate file type
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF files are supported"
        )

    # Save file to local storage
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    file_path = f"{UPLOAD_DIR}/{uuid.uuid4()}_{file.filename}"
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Create entry in Postgres with owner
    new_doc = Document(
        filename=file.filename,
        status="PENDING",
        owner_id=current_user.id
    )
    db.add(new_doc)
    await db.commit()
    await db.refresh(new_doc)

    # Trigger async Celery task
    task = process_document_task.delay(file_path, str(new_doc.id))

    # Update doc with celery task_id
    new_doc.task_id = task.id
    await db.commit()
    await db.refresh(new_doc)

    return new_doc


@router.get("", response_model=list[DocumentResponse])
async def list_documents(
    current_user: Annotated[User, Depends(get_current_active_user)],
    db: Annotated[AsyncSession, Depends(get_db)]
):
    """List all documents for the current user."""
    result = await db.execute(
        select(Document).where(Document.owner_id == current_user.id)
    )
    return result.scalars().all()


@router.get("/{document_id}", response_model=DocumentResponse)
async def get_document(
    document_id: uuid.UUID,
    current_user: Annotated[User, Depends(get_current_active_user)],
    db: Annotated[AsyncSession, Depends(get_db)]
):
    """
    Get a specific document by ID.

    Only returns documents owned by the current user.
    """
    result = await db.execute(
        select(Document).where(
            Document.id == document_id,
            Document.owner_id == current_user.id
        )
    )
    document = result.scalar_one_or_none()

    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found"
        )

    return document


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(
    document_id: uuid.UUID,
    current_user: Annotated[User, Depends(get_current_active_user)],
    db: Annotated[AsyncSession, Depends(get_db)]
):
    """
    Delete a document by ID.

    Only deletes documents owned by the current user.
    """
    result = await db.execute(
        select(Document).where(
            Document.id == document_id,
            Document.owner_id == current_user.id
        )
    )
    document = result.scalar_one_or_none()

    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found"
        )

    await db.delete(document)
    await db.commit()
