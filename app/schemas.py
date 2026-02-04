from pydantic import BaseModel, EmailStr
from uuid import UUID
from datetime import datetime


# --- User Schemas ---
class UserCreate(BaseModel):
    email: EmailStr
    password: str


class UserResponse(BaseModel):
    id: UUID
    email: str
    is_active: bool
    created_at: datetime

    class Config:
        from_attributes = True


# --- Auth Schemas ---
class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"


class TokenData(BaseModel):
    user_id: str | None = None


# --- Document Schemas ---
class DocumentResponse(BaseModel):
    id: UUID
    filename: str
    status: str
    task_id: str | None
    created_at: datetime
    owner_id: UUID | None

    class Config:
        from_attributes = True


class QuestionRequest(BaseModel):
    question: str


class AnswerResponse(BaseModel):
    question: str
    answer: str
