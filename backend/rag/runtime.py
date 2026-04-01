from functools import lru_cache

from rag.profiles import get_active_profile
from rag.service import RAGService


@lru_cache(maxsize=1)
def get_rag_service() -> RAGService:
    return RAGService(profile=get_active_profile())
