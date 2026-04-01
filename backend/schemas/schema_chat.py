from pydantic import BaseModel
from typing import Optional


class ChatRequest(BaseModel):
    query: str

class ChatResponse(BaseModel):
    answer: str
    sources: Optional[list[str]] = []
    confidence: Optional[float] = None
