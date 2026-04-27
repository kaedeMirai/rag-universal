from fastapi import APIRouter, HTTPException
from rag.runtime import get_rag_service
from rag.types import RetrievalFilters

from schemas.schema_chat import ChatRequest, ChatResponse


router = APIRouter()

@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    try:
        filters = None
        if request.filters is not None:
            filters = RetrievalFilters(
                domains=tuple(request.filters.domains),
                departments=tuple(request.filters.departments),
                doc_types=tuple(request.filters.doc_types),
                extensions=tuple(request.filters.extensions),
                languages=tuple(request.filters.languages),
                acl_groups=tuple(request.filters.acl_groups),
                created_from=request.filters.created_from,
                created_to=request.filters.created_to,
            )

        result = get_rag_service().chat(request.query, filters=filters)
        return ChatResponse(
            answer=result.answer,
            sources=result.sources,
            confidence=result.confidence,
        )
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))
