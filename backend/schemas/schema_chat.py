from datetime import datetime
from pydantic import BaseModel, Field
from typing import Optional


class ChatRetrievalFilters(BaseModel):
    domains: list[str] = Field(default_factory=list)
    departments: list[str] = Field(default_factory=list)
    doc_types: list[str] = Field(default_factory=list)
    extensions: list[str] = Field(default_factory=list)
    languages: list[str] = Field(default_factory=list)
    acl_groups: list[str] = Field(default_factory=list)
    created_from: datetime | None = None
    created_to: datetime | None = None


class ChatRequest(BaseModel):
    query: str
    filters: ChatRetrievalFilters | None = None

class ChatResponse(BaseModel):
    answer: str
    sources: Optional[list[str]] = []
    confidence: Optional[float] = None
