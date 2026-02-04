"""RAG query routes for asking questions about documents."""

from typing import Annotated

from fastapi import APIRouter, Depends

from ..models import User
from ..schemas import AnswerResponse
from ..security import get_current_active_user
from ..query import ask_question as query_documents

router = APIRouter(prefix="/query", tags=["RAG Query"])


@router.get("/ask", response_model=AnswerResponse)
async def ask(
    question: str,
    _current_user: Annotated[User, Depends(get_current_active_user)]
):
    """
    Ask a question about uploaded documents.

    - Uses semantic search to find relevant document chunks
    - Generates answer using GPT-4o with retrieved context
    - Results are cached for similar questions

    **Example questions:**
    - "What is this document about?"
    - "Summarize the key findings"
    - "What are the main recommendations?"
    """
    return query_documents(question)
