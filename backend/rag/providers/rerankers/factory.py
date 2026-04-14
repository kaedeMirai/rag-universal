from settings import settings
from rag.providers.rerankers.huggingface import HuggingFaceRerankerProvider


def create_reranker_provider():
    if settings.reranker_provider == "huggingface":
        return HuggingFaceRerankerProvider(settings.reranker_model_name)

    raise ValueError(f"Unsupported reranker provider: {settings.reranker_provider}")
