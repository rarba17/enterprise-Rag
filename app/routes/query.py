"""RAG query routes for asking questions about documents."""

import json
from typing import Annotated

from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect, status
from jose import JWTError, jwt

from ..models import User
from ..schemas import AnswerResponse
from ..security import get_current_active_user, SECRET_KEY, ALGORITHM
from ..query import ask_question as query_documents, debug_retrieval, ask_question_stream

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


@router.get("/debug")
async def debug_chunks(
    question: str,
    _current_user: Annotated[User, Depends(get_current_active_user)]
):
    """
    Debug endpoint to see what chunks are being retrieved for a question.
    Shows the actual content and similarity scores.
    """
    return debug_retrieval(question)


@router.websocket("/ws")
async def websocket_query(websocket: WebSocket, token: str = None):
    """
    WebSocket endpoint for streaming RAG responses.

    Connect with: ws://localhost:8000/query/ws?token=<your_jwt_token>

    Send JSON messages: {"question": "your question here"}
    Receive streaming response chunks, then {"done": true} when complete.
    """
    # Authenticate via token query parameter
    if not token:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION, reason="Missing token")
        return

    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id = payload.get("sub")
        if not user_id:
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION, reason="Invalid token")
            return
    except JWTError:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION, reason="Invalid token")
        return

    await websocket.accept()

    try:
        while True:
            # Receive question from client
            data = await websocket.receive_json()
            question = data.get("question")

            if not question:
                await websocket.send_json({"error": "Missing 'question' field"})
                continue

            # Stream the response
            try:
                async for chunk in ask_question_stream(question):
                    await websocket.send_json({"chunk": chunk})

                # Signal completion
                await websocket.send_json({"done": True})

            except Exception as e:
                await websocket.send_json({"error": str(e)})

    except WebSocketDisconnect:
        pass
