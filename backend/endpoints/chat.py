from fastapi import APIRouter, HTTPException
from rag.runtime import get_rag_service

from schemas.schema_chat import ChatRequest, ChatResponse


router = APIRouter()

@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    try:
        result = get_rag_service().chat(request.query)
        return ChatResponse(
            answer=result.answer,
            sources=result.sources,
            confidence=result.confidence,
        )
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))
