"""
Pydantic models — request / response schemas
"""

from pydantic import BaseModel, Field, validator
from typing import Optional, List
from enum import Enum


class MessageRole(str, Enum):
    user      = "user"
    assistant = "assistant"
    system    = "system"


class ConversationMessage(BaseModel):
    role:    MessageRole
    content: str


class ChatRequest(BaseModel):
    query:       str = Field(..., min_length=1, max_length=2000,
                             description="User's question")
    session_id:  Optional[str] = Field(None, description="Session ID for multi-turn")
    history:     Optional[List[ConversationMessage]] = Field(
                     default_factory=list,
                     description="Prior conversation turns")
    top_k:       int = Field(default=3, ge=1, le=10,
                             description="Number of context chunks to retrieve")

    @validator("query")
    def query_not_empty(cls, v):
        if not v.strip():
            raise ValueError("Query cannot be blank or whitespace only")
        return v.strip()


class SourceChunk(BaseModel):
    content:    str
    source:     str
    page:       Optional[int] = None
    score:      float = Field(..., description="Cosine similarity score")


class ChatResponse(BaseModel):
    answer:      str
    sources:     List[SourceChunk] = []
    session_id:  Optional[str] = None
    is_blocked:  bool            = False
    block_reason: Optional[str] = None
    confidence:  float           = 0.0


class DocumentUploadResponse(BaseModel):
    message:    str
    chunks_indexed: int
    filename:   str


class HealthResponse(BaseModel):
    status:  str
    version: str
    vector_store_ready: bool
